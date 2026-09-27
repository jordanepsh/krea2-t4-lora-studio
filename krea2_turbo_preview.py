#!/usr/bin/env python3
"""One-checkpoint compact NF4 Turbo preview worker for the Krea 2 Studio notebook."""
import argparse
import gc
import hashlib
import json
import os
import secrets
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatchcase
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import BitsAndBytesConfig
from diffusers import AutoencoderKLQwenImage, Krea2Pipeline, Krea2Transformer2DModel

TURBO_MODEL = "OzzyGT/Krea_2_Turbo_bnb_nf4"
TURBO_MODEL_REVISION = "5458debf8356a6646a5aa814de28dcea881f8a6d"
TRAIN_MODEL = "krea/Krea-2-Raw"
PREVIEW_MODEL_MIN_BYTES = 8 * 1024**3
PREVIEW_DISK_RESERVE_BYTES = 2 * 1024**3


class IncompletePreviewSnapshot(RuntimeError):
    pass


def validate_snapshot(snapshot, *, component):
    root = Path(snapshot)
    if component == "Turbo":
        required = (root / "model_index.json", root / "scheduler" / "scheduler_config.json",
                    root / "transformer" / "config.json")
        weight_root = root / "transformer"
    else:
        required = (root / "vae" / "config.json",)
        weight_root = root / "vae"
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    weights = [path for path in weight_root.rglob("*.safetensors") if path.is_file() and path.stat().st_size > 0]
    if missing or not weights:
        details = ", ".join(missing + ([] if weights else [f"{weight_root.relative_to(root)}/*.safetensors"]))
        raise IncompletePreviewSnapshot(f"{component} snapshot is incomplete at {root}: missing {details}")
    return root


def expected_snapshot_bytes(repo_id, revision, allow_patterns, token):
    from huggingface_hub import HfApi

    info_args = {"repo_id": repo_id, "files_metadata": True}
    if revision:
        info_args["revision"] = revision
    siblings = HfApi(token=token).model_info(**info_args).siblings
    selected = [item for item in siblings
                if any(fnmatchcase(item.rfilename, pattern) for pattern in allow_patterns)]
    if not selected:
        return None
    sizes = []
    for item in selected:
        size = getattr(item, "size", None)
        if size is None:
            size = getattr(getattr(item, "lfs", None), "size", None)
        if size is None:
            return None
        sizes.append(int(size))
    return sum(sizes)


def resolve_snapshot(repo_id, *, revision=None, allow_patterns, token, component, min_free_bytes):
    request = {"repo_id": repo_id, "allow_patterns": allow_patterns, "token": token}
    if revision:
        request["revision"] = revision
    try:
        snapshot = snapshot_download(**request, local_files_only=True)
        return validate_snapshot(snapshot, component=component)
    except (LocalEntryNotFoundError, IncompletePreviewSnapshot) as local_error:
        try:
            expected = expected_snapshot_bytes(repo_id, revision, allow_patterns, token)
        except Exception as exc:
            raise RuntimeError(
                f"{component} preview cache is incomplete, and Hugging Face could not confirm the download size. "
                "Prepare the preview cache again when the connection is available."
            ) from exc
        if expected is None:
            raise RuntimeError(
                f"{component} preview cache is incomplete, and the download size could not be verified. "
                "Prepare the preview cache again before training."
            ) from local_error
        required = max(int(min_free_bytes), expected) + PREVIEW_DISK_RESERVE_BYTES
        free = shutil.disk_usage("/content").free
        if free < required:
            raise RuntimeError(
                f"{component} preview files are missing from the local cache and cannot be downloaded safely: "
                f"{free / 1024**3:.1f} GiB free, {required / 1024**3:.1f} GiB required including reserve. "
                "Free Colab disk space, then prepare the preview cache again."
            ) from local_error
        print(f"{component} snapshot is missing or incomplete; downloading {expected / 1024**3:.1f} GiB of pinned files once ({free / 1024**3:.1f} GiB free)...", flush=True)
        snapshot = snapshot_download(**request, local_files_only=False)
        return validate_snapshot(snapshot, component=component)


def write_manifest(path, states):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema": 1, "slots": states}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def mark_failed(manifest_path, exc):
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        states = value.get("slots", {})
        for state in states.values():
            if state.get("status") in {"loading", "generating"}:
                state.update({"status": "error", "updated_at": time.time(),
                              "error": f"{type(exc).__name__}: {exc}"[:400]})
        write_manifest(manifest_path, states)
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def save_preview_files(image, version, latest):
    """Atomically publish a PNG and update latest with a hard link when supported."""
    started = time.monotonic()
    version = Path(version)
    latest = Path(latest)
    version.parent.mkdir(parents=True, exist_ok=True)
    temporary_version = version.with_name(f".{version.stem}.{os.getpid()}.tmp.png")
    temporary_latest = latest.with_name(f".{latest.stem}.{os.getpid()}.tmp.png")
    try:
        image.save(temporary_version, format="PNG", compress_level=1)
        os.replace(temporary_version, version)
        try:
            os.link(version, temporary_latest)
        except OSError:
            shutil.copy2(version, temporary_latest)
        os.replace(temporary_latest, latest)
    finally:
        for temporary in (temporary_version, temporary_latest):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return time.monotonic() - started


def main(args):
    started = time.monotonic()
    preview_root = Path(args.output) / "previews"
    manifest_path = preview_root / "latest.json"
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    slots = [slot for slot in config.get("slots", [])
             if slot.get("enabled") and str(slot.get("prompt", "")).strip()]
    if not slots:
        print("No enabled Turbo preview cards.", flush=True)
        return

    states = {}
    for slot in slots:
        number = int(slot["slot"])
        states[str(number)] = {"slot": number, "status": "loading", "step": int(args.step),
                               "prompt": str(slot["prompt"]), "updated_at": time.time()}
    write_manifest(manifest_path, states)

    cache = Path(args.cache)
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    max_seq = int(metadata["max_seq"])
    embeddings = {}
    for slot in slots:
        prompt = str(slot["prompt"]).strip()
        payload = "caption-cache-v1" + chr(0) + "krea/Krea-2-Raw" + chr(0) + str(max_seq) + chr(0) + prompt
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        embedding_file = cache / "caption_embeddings" / f"{key}_embed.pt"
        mask_file = cache / "caption_embeddings" / f"{key}_mask.pt"
        if not embedding_file.is_file() or not mask_file.is_file():
            raise FileNotFoundError(f"Cached prompt embeddings are missing for Preview card {slot['slot']}; run Prepare cache again.")
        embedding = torch.load(embedding_file, map_location="cpu", weights_only=True)
        mask = torch.load(mask_file, map_location="cpu", weights_only=True)
        if (not torch.is_tensor(embedding) or not torch.is_tensor(mask) or
                embedding.ndim != 4 or mask.ndim != 2 or embedding.shape[0] != 1 or mask.shape[0] != 1 or
                embedding.shape[1] != mask.shape[1]):
            raise ValueError(f"Cached prompt embeddings are invalid for Preview card {slot['slot']}.")
        embedding = embedding.to(torch.float16).contiguous()
        mask = mask.bool().contiguous()
        try:
            embedding = embedding.pin_memory()
            mask = mask.pin_memory()
        except RuntimeError:
            pass
        embeddings[int(slot["slot"])] = (embedding, mask)

    print(f"Loading compact NF4 Turbo for checkpoint {args.step}...", flush=True)
    model_started = time.monotonic()
    download_started = time.monotonic()
    snapshot = resolve_snapshot(
        TURBO_MODEL, revision=TURBO_MODEL_REVISION,
        allow_patterns=["model_index.json", "scheduler/*", "transformer/*"],
        token=os.environ.get("HF_TOKEN"), component="Turbo", min_free_bytes=PREVIEW_MODEL_MIN_BYTES)
    print(f"TIMING preview_download={time.monotonic() - download_started:.2f}s", flush=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_use_double_quant=False)
    transformer_started = time.monotonic()
    transformer = Krea2Transformer2DModel.from_pretrained(
        snapshot, subfolder="transformer", quantization_config=quantization,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0})
    print(f"TIMING preview_nf4_load={time.monotonic() - transformer_started:.2f}s", flush=True)
    vae_download_started = time.monotonic()
    vae_snapshot = resolve_snapshot(
        TRAIN_MODEL, allow_patterns=["vae/*"], token=os.environ.get("HF_TOKEN"),
        component="Krea 2 VAE", min_free_bytes=1024**3)
    print(f"TIMING preview_vae_download={time.monotonic() - vae_download_started:.2f}s", flush=True)
    vae_started = time.monotonic()
    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_snapshot, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print(f"TIMING preview_vae_load={time.monotonic() - vae_started:.2f}s", flush=True)
    pipeline_started = time.monotonic()
    pipe = Krea2Pipeline.from_pretrained(
        snapshot, tokenizer=None, text_encoder=None, transformer=transformer, vae=vae,
        torch_dtype=torch.float16, low_cpu_mem_usage=True)
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"TIMING preview_pipeline_load={time.monotonic() - pipeline_started:.2f}s", flush=True)
    adapter_started = time.monotonic()
    pipe.load_lora_weights(str(Path(args.adapter).parent), weight_name=Path(args.adapter).name,
                           adapter_name="studio_preview")
    print(f"TIMING preview_adapter_load={time.monotonic() - adapter_started:.2f}s", flush=True)
    print(f"TIMING preview_model_load={time.monotonic() - model_started:.2f}s", flush=True)

    failures = []
    image_seconds_total = 0.0
    image_save_work_total = 0.0
    save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="krea-preview-save")
    pending_saves = {}

    def finish_preview_save(number, wait):
        nonlocal image_save_work_total
        future, state, image_seconds, filename = pending_saves[number]
        if not wait and not future.done():
            return False
        try:
            save_seconds = future.result()
        except Exception as exc:
            failures.append((number, str(exc)))
            state.update({"status": "error", "updated_at": time.time(),
                          "error": f"Image save failed: {type(exc).__name__}: {exc}"[:400]})
            print(f"PREVIEW_ERROR slot={number}: image save failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            image_save_work_total += save_seconds
            state.update({"status": "ready", "updated_at": time.time(), "filename": filename,
                          "seconds": image_seconds, "save_seconds": save_seconds})
            print(f"TIMING preview_save_{number}={save_seconds:.2f}s", flush=True)
        states[str(number)] = state
        write_manifest(manifest_path, states)
        del pending_saves[number]
        return True

    try:
        for index, slot in enumerate(slots, 1):
            for saved_number in list(pending_saves):
                finish_preview_save(saved_number, wait=False)
            number = int(slot["slot"])
            prompt = str(slot["prompt"]).strip()
            seed = secrets.randbelow(2**32) if slot.get("random_seed") else int(slot.get("seed", 42))
            state = {"slot": number, "status": "generating", "step": int(args.step),
                     "prompt": prompt, "seed": seed, "updated_at": time.time()}
            states[str(number)] = state
            write_manifest(manifest_path, states)
            print(f"preview {index}/{len(slots)} · slot {number} · step {args.step} · seed {seed}", flush=True)
            image_started = time.monotonic()
            generated = prompt_embeds = prompt_mask = generator = None
            try:
                prompt_embeds, prompt_mask = embeddings[number]
                generator = torch.Generator(device="cuda").manual_seed(seed)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    generated = pipe(prompt_embeds=prompt_embeds.to("cuda", non_blocking=True),
                                     prompt_embeds_mask=prompt_mask.to("cuda", non_blocking=True),
                                     height=int(args.resolution), width=int(args.resolution),
                                     num_inference_steps=8, guidance_scale=0.0,
                                     generator=generator).images[0]
            except Exception as exc:
                image_seconds_total += time.monotonic() - image_started
                failures.append((number, str(exc)))
                del generated, prompt_embeds, prompt_mask, generator
                state.update({"status": "error", "updated_at": time.time(), "error": str(exc)[:400]})
                states[str(number)] = state
                write_manifest(manifest_path, states)
                print(f"PREVIEW_ERROR slot={number}: {type(exc).__name__}: {exc}", flush=True)
                continue

            image_seconds = time.monotonic() - image_started
            image_seconds_total += image_seconds
            folder = preview_root / f"slot_{number}"
            folder.mkdir(parents=True, exist_ok=True)
            version = folder / f"step_{int(args.step):05d}.png"
            latest = folder / "latest.png"
            future = save_executor.submit(save_preview_files, generated, version, latest)
            pending_saves[number] = (future, state, image_seconds, version.name)
            state.update({"status": "saving", "updated_at": time.time(),
                          "seconds": image_seconds, "filename": version.name})
            states[str(number)] = state
            write_manifest(manifest_path, states)
            del generated, prompt_embeds, prompt_mask, generator
            print(f"TIMING preview_image_{number}={image_seconds:.2f}s", flush=True)
        for saved_number in list(pending_saves):
            finish_preview_save(saved_number, wait=True)
    finally:
        try:
            for saved_number in list(pending_saves):
                finish_preview_save(saved_number, wait=True)
        finally:
            save_executor.shutdown(wait=True)
            print(f"TIMING preview_images_total={image_seconds_total:.2f}s", flush=True)
            print(f"TIMING preview_save_work_total={image_save_work_total:.2f}s", flush=True)
            print(f"PREVIEW_PROCESS_TIME={time.monotonic() - started:.2f}s", flush=True)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception as exc:
                    print(f"GPU synchronize during preview cleanup: {type(exc).__name__}: {exc}", flush=True)
            del pipe, transformer, vae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        print(f"{len(failures)} preview card(s) failed; saved checkpoints remain valid.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        manifest = Path(args.output) / "previews" / "latest.json"
        mark_failed(manifest, exc)
        print(f"TURBO_PREVIEW_FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise
