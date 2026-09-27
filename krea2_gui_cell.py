import gc
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from IPython.display import HTML, display
from PIL import Image
from huggingface_hub import HfApi
from safetensors import safe_open

PREVIEW_MODEL_REPO = "OzzyGT/Krea_2_Turbo_bnb_nf4"
PREVIEW_MODEL_REVISION = "5458debf8356a6646a5aa814de28dcea881f8a6d"
PREVIEW_MODEL_PATTERNS = ["model_index.json", "scheduler/*", "transformer/*"]
PREVIEW_MODEL_MIN_BYTES = 8 * 1024**3
PREVIEW_DISK_RESERVE_BYTES = 2 * 1024**3


def validate_preview_snapshot(snapshot):
    root = Path(snapshot)
    required = (root / "model_index.json", root / "scheduler" / "scheduler_config.json",
                root / "transformer" / "config.json")
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    weights = [path for path in (root / "transformer").rglob("*.safetensors")
               if path.is_file() and path.stat().st_size > 0]
    if missing or not weights:
        details = ", ".join(missing + ([] if weights else ["transformer/*.safetensors"]))
        raise RuntimeError(f"Turbo preview snapshot is incomplete: missing {details}.")
    if root.name != PREVIEW_MODEL_REVISION:
        raise RuntimeError(f"Turbo preview snapshot revision mismatch: expected {PREVIEW_MODEL_REVISION}, got {root.name}.")
    return root


def build_train_command(config, cache, output, resume=False):
    command = ["/content/krea2_t4_train.py", "--cache", str(cache), "--output", str(output),
               "--steps", str(config["steps"]), "--rank", str(config["rank"]),
               "--lr", str(config["lr"]), "--seed", str(config["seed"]),
               "--save-every", str(min(config["save_every"], config["steps"])),
               "--keep-checkpoints", str(config["keep"]), "--grad-accum", "1"]
    if config["grad_ckpt"]:
        command.append("--grad-ckpt")
    if config["attention_only"]:
        command.append("--attention-only")
    if config.get("preview_enabled"):
        command.append("--preview-at-checkpoints")
    if resume:
        command.append("--resume")
    return command


def build_preview_command(config, cache, adapter, output, preview_config, step):
    return ["/content/krea2_t4_preview.py", "--cache", str(cache),
            "--adapter", str(adapter), "--output", str(output),
            "--config", str(preview_config), "--step", str(int(step)),
            "--resolution", str(int(config["preview_resolution"]))]


class GpuMemoryReleaseError(RuntimeError):
    """A preview child kept enough VRAM allocated to endanger training resume."""


class Field:
    def __init__(self, value=None):
        self.value = value
        self.options = []
        self.disabled = False


def elapsed_label(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"


class KreaStudio:
    """HTTP-backed UI state around the existing process-isolated training stages."""

    def __init__(self):
        self.closed = threading.Event()
        self.busy = False
        self.dataset_dir = None
        self.run_label = "Ready"
        self.log_lines = []
        self.last_progress = (-1, "")
        self.last_result = None
        self.timings = {"setup": globals().get("KREA_SETUP_SECONDS")}
        self.preview_prefetch_result = None
        self.stage_started = None
        self.current_stage = None
        self.cache_started = None
        self.vae_started = self.vae_first = self.qwen_started = self.qwen_first = None
        self.train_started = self.train_first = None
        self.last_step_at = None
        self.last_step = self.train_target = None
        self.sec_per_step = self.eta_seconds = None
        self.resource_updated_at = 0.0

        defaults = {"project":"my_krea2_lora", "trigger":"my_trigger_word", "caption":"character appearance and style",
            "use_txt":True, "test_mode":False, "resolution":512, "steps":500, "rank":16,
            "lr":0.0003, "save_every":50, "max_seq":128, "attention_only":True,
            "grad_ckpt":True, "seed":42, "preview_enabled":False,
            "preview_1_enabled":True, "preview_1_prompt":"", "preview_1_seed":42, "preview_1_random":False,
            "preview_2_enabled":False, "preview_2_prompt":"", "preview_2_seed":43, "preview_2_random":False,
            "preview_3_enabled":False, "preview_3_prompt":"", "preview_3_seed":44, "preview_3_random":False,
            "preview_4_enabled":False, "preview_4_prompt":"", "preview_4_seed":45, "preview_4_random":False,
            "preview_resolution":512, "preview_cleanup_cache":True,
            "push":False, "hub_id":"", "private":True, "keep":2, "drive":False, "token":""}
        for name, value in defaults.items():
            setattr(self, name, Field(value))
        self.dataset_path = Field("")
        for name in ("status", "resources", "progress", "dataset_summary", "results", "log"):
            setattr(self, name, Field(""))
        for name in ("upload_button", "path_button", "cache_button", "train_button", "resume_button", "sync_button"):
            setattr(self, name, Field())
        self._set_status("Ready", "Choose a dataset to begin.")
        self._set_progress(0, "Waiting for dataset")
        self._render_log()
        self._render_resources()
        self.refresh_results()

    def close(self):
        self.closed.set()

    def _root(self):
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.project.value.strip()).strip("._")
        if not name:
            raise ValueError("Enter a project name.")
        if self.drive.value:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
            return Path("/content/drive/MyDrive/Krea2_Colab") / name
        return Path("/content/Krea2_Colab") / name

    def _config(self):
        steps = min(self.steps.value, 20) if self.test_mode.value else self.steps.value
        rank = min(self.rank.value, 8) if self.test_mode.value else self.rank.value
        if not 0 < self.lr.value < .01:
            raise ValueError("Learning rate must be between 0 and 0.01.")
        if not self.trigger.value.strip():
            raise ValueError("Enter a trigger word.")
        save_every = max(1, min(int(self.save_every.value), int(steps)))
        preview_enabled = bool(self.preview_enabled.value)
        preview_resolution = int(self.preview_resolution.value)
        if preview_resolution not in (512, 768, 1024):
            raise ValueError("Turbo preview size must be 512, 768, or 1024 pixels.")
        preview_slots = []
        if preview_enabled:
            for slot in range(1, 5):
                if not getattr(self, f"preview_{slot}_enabled").value:
                    continue
                prompt = str(getattr(self, f"preview_{slot}_prompt").value).strip()
                if not prompt:
                    raise ValueError(f"Enter a prompt for Preview card {slot}, or turn that card off.")
                if len(prompt) > 2000:
                    raise ValueError(f"Preview card {slot} prompt must be 2,000 characters or fewer.")
                seed = int(getattr(self, f"preview_{slot}_seed").value)
                if not 0 <= seed <= 4294967295:
                    raise ValueError(f"Preview card {slot} seed must be between 0 and 4,294,967,295.")
                preview_slots.append({"slot":slot, "enabled":True, "prompt":prompt, "seed":seed,
                                      "random_seed":bool(getattr(self, f"preview_{slot}_random").value)})
            if not preview_slots:
                raise ValueError("Enable at least one Preview card, or turn preview generation off.")
        return {"steps": steps, "rank": rank, "resolution": self.resolution.value,
                "max_seq": self.max_seq.value, "lr": self.lr.value, "save_every": save_every,
                "seed": self.seed.value, "attention_only": self.attention_only.value,
                "grad_ckpt": self.grad_ckpt.value, "preview_enabled":preview_enabled,
                "preview_resolution":preview_resolution,
                "preview_cleanup_cache":bool(self.preview_cleanup_cache.value),
                "preview_slots":preview_slots,
                "push": self.push.value, "private": self.private.value, "keep": self.keep.value,
                "hub_id": self.hub_id.value.strip(), "drive": self.drive.value,
                "trigger": self.trigger.value.strip(), "caption": self.caption.value.strip(),
                "use_txt": self.use_txt.value}

    def _token(self):
        value = self.token.value.strip()
        if not value:
            try:
                from google.colab import userdata
                value = (userdata.get("HF_TOKEN") or "").strip()
            except Exception:
                pass
        if not value:
            raise ValueError("Set HF_TOKEN in Colab Secrets or enter a token above.")
        return value

    def _repo_id(self, token):
        if self.hub_id.value.strip():
            return self.hub_id.value.strip()
        username = HfApi(token=token).whoami()["name"]
        return f"{username}/{self._root().name}"

    def _set_status(self, title, detail=""):
        self.run_label = title
        self.status.value = ("<div style='border:2px solid #000;padding:16px;background:#fff;color:#000'>"
                             f"<b style='font-size:18px'>{html.escape(title)}</b><br>{html.escape(detail)}</div>")

    def _set_progress(self, pct, label):
        pct = max(0, min(100, int(pct)))
        if (pct, label) == self.last_progress:
            return
        self.last_progress = (pct, label)
        self.progress.value = ("<div style='padding:10px 0;color:#000'><b>Progress</b> · "
                               f"{html.escape(label)} · {pct}%<div style='height:14px;border:1px solid #000;background:#fff'>"
                               f"<div style='height:14px;width:{pct}%;background:#000'></div></div></div>")

    def _append_log(self, line):
        line = line.strip()
        if not line or line.startswith("Loading weights:") or line.startswith("Fetching"):
            return
        if line.startswith("/usr/local/lib/") or "FutureWarning" in line:
            return
        self.log_lines.append(line[:600])
        self.log_lines = self.log_lines[-80:]
        self._render_log()

    def _render_log(self):
        lines = "\n".join(self.log_lines[-24:]) or "Stage messages will appear here."
        self.log.value = ("<div style='border:1px solid #000;padding:12px;background:#000;color:#fff'>"
                          "<b>Activity</b><pre style='white-space:pre-wrap;max-height:260px;overflow:auto;color:#fff'>"
                          f"{html.escape(lines)}</pre></div>")

    def _render_resources(self):
        try:
            import psutil
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            disk = shutil.disk_usage("/content")
            query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
            gpu = query.stdout.strip().splitlines()[0] if query.returncode == 0 and query.stdout.strip() else "unavailable"
            if gpu != "unavailable":
                used, total, util = [v.strip() for v in gpu.split(",")]
                gpu = f"{int(used)/1024:.1f} / {int(total)/1024:.1f} GB · {util}% busy"
            self.resource_updated_at = time.time()
            self.resources.value = ("<div><b>LIVE SYSTEM</b> · "
                f"CPU {cpu:.0f}% · RAM {ram.used/2**30:.1f}/{ram.total/2**30:.1f} GB"
                f" · GPU {html.escape(gpu)} · Free disk {disk.free/2**30:.1f} GB"
                f"<small style='display:block;color:#888;margin-top:4px'>Sampled {time.strftime('%H:%M:%S')}</small></div>")
        except Exception as e:
            self.resources.value = f"<div>System metrics unavailable: {html.escape(str(e))}</div>"

    def _resource_loop(self):
        while not self.closed.is_set():
            self._render_resources()
            self.closed.wait(2 if self.busy else 6)

    def _set_busy(self, busy):
        self.busy = busy
        for button in (self.upload_button, self.path_button, self.cache_button, self.train_button,
                       self.resume_button, self.sync_button):
            button.disabled = busy

    def _start(self, name, fn):
        if self.busy:
            return
        self._set_busy(True)
        self.current_stage = name
        self._set_status(name.title(), "Stage running. Keep this Colab tab open.")
        self._set_progress(0, "Starting")
        if name in ("train", "resume"):
            self.train_first = None
            self.last_step_at = None
            self.last_step = self.train_target = None
            self.sec_per_step = None
            self.eta_seconds = None
            self.timings.pop("train_cold_start", None)
            self.timings.pop("train_total", None)
            self.timings.pop("train_loop_total", None)
            self.timings.pop("preview_total", None)
        if name == "cache":
            self.vae_started = self.vae_first = self.qwen_started = self.qwen_first = None
            for key in ("vae_load", "vae_encode", "vae_total", "vae_stage", "qwen_load", "caption_encode", "cache_total"):
                self.timings.pop(key, None)
        try:
            fn()
            self._set_status("Complete", f"{name.title()} completed successfully.")
            self._set_progress(100, "Complete")
        except Exception as exc:
            self.last_result = exc
            self._append_log(f"ERROR: {type(exc).__name__}: {exc}")
            self._set_status("Needs attention", str(exc))
        finally:
            self._set_busy(False)
            self.refresh_results()
            self._render_resources()

    def _new_dataset_dir(self):
        path = self._root() / f"dataset_{int(time.time())}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _ingest_files(self, files):
        allowed = {".zip", ".png", ".jpg", ".jpeg", ".webp", ".txt"}
        out = self._new_dataset_dir()
        count = 0
        for name, data in files:
            if Path(name).suffix.lower() not in allowed:
                continue
            if Path(name).suffix.lower() == ".zip":
                import io
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    count += self._extract_archive(archive, out)
            else:
                target = out / Path(name).name
                if target.exists():
                    raise ValueError(f"Duplicate filename: {target.name}")
                target.write_bytes(data)
                count += 1
        self._validate_dataset(out)
        self.dataset_dir = out
        self._append_log(f"Dataset ready: {out} ({count} files)")
        self._set_status("Dataset ready", f"{len(self._images(out))} images. Next: Prepare cache.")

    def _extract_archive(self, archive, out):
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".txt"}
        infos = archive.infolist()
        if len(infos) > 2000 or sum(i.file_size for i in infos) > 5 * 2**30:
            raise ValueError("ZIP is too large or contains too many entries.")
        count = 0
        for entry in infos:
            if entry.is_dir() or Path(entry.filename).suffix.lower() not in allowed:
                continue
            target = out / Path(entry.filename).name
            if target.exists():
                raise ValueError(f"Duplicate filename: {target.name}")
            with archive.open(entry) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            count += 1
        return count

    def _images(self, directory):
        return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})

    def _validate_dataset(self, directory):
        images = self._images(directory)
        if not images:
            raise ValueError("No JPG, PNG or WEBP images found.")
        for image in images:
            with Image.open(image) as opened:
                opened.verify()
        captions = sum(p.with_suffix(".txt").exists() for p in images)
        self.dataset_summary.value = ("<div style='border:1px solid #000;padding:12px;color:#000'>"
            f"<b>Dataset</b> · {len(images)} valid images · {captions} matching captions<br>"
            f"<small>{html.escape(str(directory))}</small></div>")

    def _use_upload(self):
        if self.busy:
            return
        try:
            selected = self.upload.value
            items = list(selected.values()) if isinstance(selected, dict) else list(selected)
            if not items:
                raise ValueError("Choose a ZIP or image files first.")
            pairs = [(item["name"], bytes(item["content"])) for item in items]
            self._ingest_files(pairs)
            self.upload.value = ()
        except Exception as exc:
            self._set_status("Dataset error", str(exc))

    def _use_path(self):
        if self.busy:
            return
        try:
            path = Path(self.dataset_path.value.strip()).expanduser()
            if path.is_dir():
                self._validate_dataset(path)
                self.dataset_dir = path
                self._set_status("Dataset ready", f"{len(self._images(path))} images. Next: Prepare cache.")
            elif path.is_file() and path.suffix.lower() == ".zip":
                out = self._new_dataset_dir()
                with zipfile.ZipFile(path) as archive:
                    count = self._extract_archive(archive, out)
                self._validate_dataset(out)
                self.dataset_dir = out
                self._append_log(f"Dataset ready: {out} ({count} files)")
                self._set_status("Dataset ready", f"{len(self._images(out))} images. Next: Prepare cache.")
            else:
                raise ValueError("Enter an existing ZIP file or folder path in Colab.")
        except Exception as exc:
            self._set_status("Dataset error", str(exc))

    def _cache_signature(self, config):
        if self.dataset_dir is None:
            raise ValueError("Load a dataset first.")
        files = sorted(p for p in self.dataset_dir.iterdir() if p.suffix.lower() in
            {".png", ".jpg", ".jpeg", ".webp", ".txt"})
        payload = []
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            payload.append((path.name, digest.hexdigest()))
        preview_prompts = sorted({slot["prompt"] for slot in config.get("preview_slots", [])})
        payload.append({"model":"krea/Krea-2-Raw", "cache_format":3,
                        **{k:config[k] for k in ("resolution", "max_seq", "trigger", "caption", "use_txt")},
                        "preview_prompts":preview_prompts})
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    @staticmethod
    def _preview_embedding_paths(cache, prompt, max_seq):
        payload = f"caption-cache-v1\0krea/Krea-2-Raw\0{max_seq}\0{prompt}"
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        directory = Path(cache) / "caption_embeddings"
        return directory / f"{key}_embed.pt", directory / f"{key}_mask.pt"

    @staticmethod
    def _write_preview_config(config, path):
        payload = {"enabled":bool(config.get("preview_enabled")),
                   "slots":config.get("preview_slots", []) if config.get("preview_enabled") else []}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def _cache_complete(self, cache, signature):
        marker = cache / "studio_signature.json"
        metadata_file = cache / "metadata.json"
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("signature") != signature:
                return False
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            items = metadata.get("items", [])
            if len(items) != len(self._images(self.dataset_dir)):
                return False
            for item in items:
                for key in ("latent", "embed", "mask"):
                    artifact = cache / item[key]
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        return False
            for slot in getattr(self, "_active_preview_slots", []):
                for artifact in self._preview_embedding_paths(cache, slot["prompt"], self.max_seq.value):
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        return False
            return True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def _gpu_memory_snapshot():
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4)
            if query.returncode != 0 or not query.stdout.strip():
                return None
            values = [int(value.strip()) for value in query.stdout.strip().splitlines()[0].split(",")]
            return (values[0], values[1]) if len(values) == 2 else None
        except Exception:
            return None

    def _wait_for_gpu_release(self, before, timeout=15.0):
        if before is None:
            self.timings["preview_vram_reclaim"] = None
            self._append_log("VRAM release probe unavailable; the preview worker exited in its own process.")
            return
        started = time.monotonic()
        after = None
        threshold = before[0] + 512
        while True:
            after = self._gpu_memory_snapshot()
            if after is None or after[0] <= threshold or time.monotonic() - started >= timeout:
                break
            time.sleep(0.5)
        elapsed = time.monotonic() - started
        self.timings["preview_vram_reclaim"] = elapsed
        if after is None:
            self._append_log("VRAM release probe became unavailable after preview; process isolation completed.")
            return
        message = (f"Preview VRAM after process exit: {after[0] / 1024:.2f}/{after[1] / 1024:.2f} GiB; "
                   f"pre-preview baseline {before[0] / 1024:.2f} GiB; check {elapsed_label(elapsed)}.")
        self._append_log(message)
        if after[0] > threshold:
            raise GpuMemoryReleaseError(
                message + " Training remains at its saved checkpoint because VRAM did not return near baseline.")

    def _run_process(self, command, env, stage, total, allowed_return_codes=()):
        gpu_before = self._gpu_memory_snapshot() if stage == "preview" else None
        started = time.monotonic()
        self.stage_started = started
        self.current_stage = stage
        process = subprocess.Popen([sys.executable, "-u", *command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env, bufsize=0)
        pending = ""
        next_resource_update = 0
        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                if time.monotonic() >= next_resource_update:
                    self._render_resources()
                    next_resource_update = time.monotonic() + 2
                pending += chunk.decode("utf-8", errors="replace")
                parts = re.split(r"[\r\n]", pending)
                pending = parts.pop()
                for line in parts:
                    self._process_line(line, stage, total)
            if pending.strip():
                self._process_line(pending, stage, total)
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
        if stage == "preview":
            self._wait_for_gpu_release(gpu_before)
        elapsed = time.monotonic() - started
        timing_key = f"{stage}_total"
        self.timings[timing_key] = self.timings.get(timing_key, 0.0) + elapsed
        self._append_log(f"{stage.title()} process time: {elapsed_label(elapsed)}.")
        if return_code and return_code not in allowed_return_codes:
            raise RuntimeError(f"{stage} exited with code {return_code}; see Activity.")
        return return_code

    def _process_line(self, line, stage, total):
        line = line.strip()
        if not line:
            return
        timing = re.match(r"^TIMING\s+([a-z0-9_]+)=([\d.]+)s$", line, re.I)
        if timing:
            key, value = timing.group(1).lower(), float(timing.group(2))
            self.timings[key] = value
            if key == "train_steady_sec_per_step":
                self.sec_per_step = value
                self.timings["sec_per_step"] = value
            if key == "caption_cache":
                self.timings["caption_encode"] = value
            if key in {"qwen_download", "tokenizer_load", "qwen_model_load"}:
                self.timings["qwen_load"] = sum(self.timings.get(k, 0) for k in
                    ("qwen_download", "tokenizer_load", "qwen_model_load"))
        if stage == "cache":
            now = time.monotonic()
            if "[1/2]" in line:
                self.vae_started = now
            if "[2/2]" in line:
                self.qwen_started = now
                if self.vae_started is not None:
                    self.timings["vae_stage"] = now - self.vae_started
            vae = re.search(r"\b(\d+)/(\d+)\s+\S+\.(?:png|jpg|jpeg|webp)\s+->", line, re.I)
            caption = re.search(r"caption\s+(\d+)/(\d+)", line, re.I)
            if vae:
                if self.vae_first is None:
                    self.vae_first = now
                    if self.vae_started is not None:
                        self.timings["vae_load"] = now - self.vae_started
                n, count = int(vae.group(1)), max(1, int(vae.group(2)))
                self._set_progress(5 + 45 * n / count, f"VAE encode · {n}/{count}")
                if n >= count and self.vae_first is not None:
                    self.timings["vae_encode"] = now - self.vae_first
            elif caption:
                if self.qwen_first is None:
                    self.qwen_first = now
                    if self.qwen_started is not None and "qwen_load" not in self.timings and "reused" not in line:
                        self.timings["qwen_load"] = now - self.qwen_started
                n, count = int(caption.group(1)), max(1, int(caption.group(2)))
                self._set_progress(55 + 45 * n / count, f"Caption cache · {n}/{count}")
                if n >= count and self.qwen_first is not None:
                    self.timings["caption_encode"] = now - self.qwen_first
            elif "[2/2]" in line:
                self._set_progress(55, "Loading Qwen3-VL in NF4")
        elif stage == "preview":
            sample = re.search(r"\bpreview\s+(\d+)/(\d+)\b", line, re.I)
            if sample:
                n, count = int(sample.group(1)), max(1, int(sample.group(2)))
                self._set_progress(100 * n / count, f"Turbo preview · {n}/{count}")
            if line.startswith("TIMING "):
                self._append_log(line)
                return
        elif stage == "train":
            now = time.monotonic()
            step = re.search(r"\bstep\s+(\d+)/(\d+)\b", line, re.I)
            speed = re.search(r"([\d.]+)s/step", line, re.I)
            eta = re.search(r"ETA\s+([\d.]+)\s*([smh])", line, re.I)
            if step:
                n, count = int(step.group(1)), max(1, int(step.group(2)))
                self.last_step_at = now
                self.last_step, self.train_target = n, count
                self._set_progress(100 * n / count, f"Training · {n}/{count}")
                if self.train_first is None:
                    self.train_first = now
                    if self.stage_started is not None:
                        self.timings["train_cold_start"] = now - self.stage_started
                if speed:
                    self.sec_per_step = float(speed.group(1))
                    self.timings["sec_per_step"] = self.sec_per_step
                if eta:
                    self.eta_seconds = float(eta.group(1)) * {"s":1,"m":60,"h":3600}[eta.group(2).lower()]
                elif self.sec_per_step is not None:
                    self.eta_seconds = max(0, (count-n) * self.sec_per_step)
            elif "Loading NF4" in line:
                self._set_progress(2, "Loading Krea 2 transformer in NF4")
        self._append_log(line)

    def _prepare_cache(self):
        config = self._config()
        self._active_preview_slots = config.get("preview_slots", []) if config.get("preview_enabled") else []
        signature = self._cache_signature(config)
        root = self._root()
        cache = root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        marker = cache / "studio_signature.json"
        preview_config = self._write_preview_config(config, root / "preview_config.json")
        token = self._token() if config["preview_enabled"] else None
        prefetch_thread = None
        if config["preview_enabled"]:
            self._append_log("Compact Turbo NF4 prefetch started alongside cache preparation.")
            prefetch_thread = threading.Thread(
                target=self._prefetch_preview_model,
                args=(token, config["preview_cleanup_cache"]),
                name="krea2-turbo-prefetch", daemon=True)
            prefetch_thread.start()
        if self._cache_complete(cache, signature):
            self.timings.update({"vae_load":0.0,"vae_encode":0.0,"vae_stage":0.0,
                                 "qwen_load":0.0,"caption_encode":0.0,"cache_total":0.0})
            self._append_log("Cache is current and all latent/caption files are present; preprocessing skipped.")
            self._append_log("Cache reuse timing · VAE 0s · Qwen 0s · total 0s.")
            if prefetch_thread is not None:
                prefetch_thread.join()
                self._require_preview_prefetch_ready()
            return
        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token
        else:
            env["HF_TOKEN"] = self._token()
        self.cache_started = time.monotonic()
        command = ["/content/krea2_t4_precache.py", "--dataset", str(self.dataset_dir),
                   "--cache", str(cache), "--resolution", str(config["resolution"]),
                   "--max-seq", str(config["max_seq"]), "--trigger", config["trigger"],
                   "--default-caption", config["caption"], "--preview-config", str(preview_config)]
        if config["use_txt"]:
            command.append("--use-txt")
        try:
            self._run_process(command, env, "cache", len(self._images(self.dataset_dir)))
        finally:
            if prefetch_thread is not None:
                prefetch_thread.join()
        if prefetch_thread is not None:
            self._require_preview_prefetch_ready()
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        if len(metadata["items"]) != len(self._images(self.dataset_dir)):
            raise RuntimeError("Cache item count differs from dataset image count.")
        for item in metadata["items"]:
            if any(not (cache / item[key]).is_file() or (cache / item[key]).stat().st_size == 0
                   for key in ("latent", "embed", "mask")):
                raise RuntimeError(f"Cache artifact is missing or empty for {item.get('source', item.get('id'))}.")
        for slot in self._active_preview_slots:
            if any(not path.is_file() or path.stat().st_size == 0
                   for path in self._preview_embedding_paths(cache, slot["prompt"], config["max_seq"])):
                raise RuntimeError(f"Cached text embeddings are missing for preview card {slot['slot']}.")
        marker_tmp = marker.with_suffix(".json.tmp")
        marker_tmp.write_text(json.dumps({"signature": signature, "items": len(metadata["items"]),
                                          "verified_at": time.time()}, indent=2), encoding="utf-8")
        os.replace(marker_tmp, marker)
        self.timings["cache_total"] = time.monotonic() - self.cache_started
        self._append_log(f"Cache ready: {len(metadata['items'])} images.")
        self._append_log("Cache timing · " + " · ".join(
            f"{label}: {elapsed_label(self.timings[key])}" for label, key in
            (("VAE load", "vae_load"), ("VAE encode", "vae_encode"),
             ("Qwen load", "qwen_load"), ("caption encode", "caption_encode"),
             ("total", "cache_total")) if key in self.timings))

    def _prefetch_preview_model(self, token, replace_full_precision_cache):
        started = time.monotonic()
        self.preview_prefetch_result = {"ready": False, "error": "Turbo prefetch did not complete."}
        try:
            from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

            cache_info = scan_cache_dir()
            local_snapshot = None
            try:
                candidate = snapshot_download(
                    repo_id=PREVIEW_MODEL_REPO, revision=PREVIEW_MODEL_REVISION,
                    allow_patterns=PREVIEW_MODEL_PATTERNS, token=token, local_files_only=True)
                validate_preview_snapshot(candidate)
                local_snapshot = str(candidate)
                self._append_log("Compact NF4 Turbo cache already exists and passed integrity checks.")
            except Exception as exc:
                self._append_log(f"No complete local Turbo snapshot found; checking exact download size ({type(exc).__name__}).")

            expected_bytes = None
            if local_snapshot is None:
                model_info = HfApi(token=token).model_info(
                    PREVIEW_MODEL_REPO, revision=PREVIEW_MODEL_REVISION, files_metadata=True)
                selected = [item for item in model_info.siblings
                            if item.rfilename == "model_index.json" or item.rfilename.startswith("scheduler/")
                            or item.rfilename.startswith("transformer/")]
                sizes = []
                missing_sizes = []
                for item in selected:
                    size = getattr(item, "size", None)
                    if size is None:
                        size = getattr(getattr(item, "lfs", None), "size", None)
                    if size is None:
                        missing_sizes.append(item.rfilename)
                    else:
                        sizes.append(int(size))
                if not selected or missing_sizes:
                    detail = ", ".join(missing_sizes[:3]) or "no matching files reported"
                    raise RuntimeError(f"Hugging Face did not provide exact sizes for the pinned Turbo files ({detail}).")
                expected_bytes = sum(sizes)
                required_free = max(PREVIEW_MODEL_MIN_BYTES, expected_bytes) + PREVIEW_DISK_RESERVE_BYTES
                free = shutil.disk_usage("/content").free

                if free < required_free and replace_full_precision_cache:
                    old_repo = next((repo for repo in cache_info.repos
                                     if repo.repo_id == "krea/Krea-2-Turbo"), None)
                    revisions = [revision.commit_hash for revision in getattr(old_repo, "revisions", [])]
                    if revisions:
                        cleanup = cache_info.delete_revisions(*revisions)
                        if cleanup.expected_freed_size:
                            self._append_log(
                                "Low disk space: replacing only the re-downloadable full-precision Turbo Hugging Face cache "
                                f"({cleanup.expected_freed_size / 1024**3:.1f} GiB) with compact NF4. "
                                "Dataset, cache, checkpoints and LoRA outputs are kept.")
                            cleanup.execute()
                            free = shutil.disk_usage("/content").free

                if free < required_free:
                    self._append_log(
                        f"Compact Turbo prefetch stopped: {free / 1024**3:.1f} GiB free; "
                        f"{required_free / 1024**3:.1f} GiB is required for the exact download plus reserve. "
                        "No project data was removed; free disk space and run Prepare cache again.")
                    self.preview_prefetch_result = {
                        "ready": False,
                        "error": f"Only {free / 1024**3:.1f} GiB disk space is free; {required_free / 1024**3:.1f} GiB is required.",
                    }
                    return

                self._append_log(
                    f"Downloading pinned compact NF4 Turbo files ({expected_bytes / 1024**3:.1f} GiB; "
                    "text encoder and VAE excluded) while cache preparation runs.")
                candidate = snapshot_download(
                    repo_id=PREVIEW_MODEL_REPO, revision=PREVIEW_MODEL_REVISION,
                    allow_patterns=PREVIEW_MODEL_PATTERNS, token=token)
                validate_preview_snapshot(candidate)
                local_snapshot = str(candidate)

            self._append_log(
                f"Compact Turbo cache verified in {elapsed_label(time.monotonic() - started)}; "
                f"free disk {shutil.disk_usage('/content').free / 1024**3:.1f} GiB.")
            self.preview_prefetch_result = {"ready": True, "snapshot": local_snapshot}
        except Exception as exc:
            self._append_log(f"Turbo prefetch did not finish: {type(exc).__name__}: {exc}")
            self.preview_prefetch_result = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.timings["preview_prefetch"] = time.monotonic() - started

    def _require_preview_prefetch_ready(self):
        result = getattr(self, "preview_prefetch_result", None) or {}
        if result.get("ready"):
            return
        detail = result.get("error", "No verified Turbo cache result was recorded.")
        raise RuntimeError(
            "Turbo preview is enabled, but its model cache is not ready, so training was not started. "
            f"{detail} Disable Optional Turbo preview images to train without previews, or correct the network/disk issue and run Prepare cache again."
        )

    def _preview_complete(self, output, step, slots):
        if not slots:
            return True
        manifest = Path(output) / "previews" / "latest.json"
        try:
            states = json.loads(manifest.read_text(encoding="utf-8")).get("slots", {})
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        for slot in slots:
            state = states.get(str(slot["slot"]), {})
            image = Path(output) / "previews" / f"slot_{slot['slot']}" / "latest.png"
            if state.get("status") != "ready" or state.get("step") != int(step) or not image.is_file():
                return False
        return True

    def _run_preview(self, config, cache, output, preview_config, step, env):
        slots = config.get("preview_slots", []) if config.get("preview_enabled") else []
        if not slots:
            return
        adapter = Path(output) / "versions" / f"lora_step_{int(step):05d}.safetensors"
        if not adapter.is_file():
            checkpoint = Path(output) / f"checkpoint-{int(step)}" / "pytorch_lora_weights.safetensors"
            if checkpoint.is_file():
                adapter = checkpoint
            else:
                raise RuntimeError(f"No saved LoRA adapter found for preview step {step}.")
        command = build_preview_command(config, cache, adapter, output, preview_config, step)
        self._append_log(f"Turbo preview stage · step {step} · {len(slots)} enabled card(s).")
        self._run_process(command, env, "preview", len(slots))

    def _train(self, resume):
        config = self._config()
        root = self._root()
        cache = root / "cache"
        output = root / "output"
        manifest_path = root / "run_manifest.json"

        if self.dataset_dir is None and resume and manifest_path.is_file():
            try:
                saved_path = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_path", "")
                if saved_path and Path(saved_path).is_dir():
                    self._validate_dataset(Path(saved_path))
                    self.dataset_dir = Path(saved_path)
                    self._append_log(f"Restored dataset selection from run manifest: {self.dataset_dir}")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if self.dataset_dir is None:
            raise ValueError("Choose the dataset again, or use Drive storage so the saved run can restore it after a runtime reset.")

        marker = cache / "studio_signature.json"
        if not (cache / "metadata.json").exists() or not marker.exists():
            raise ValueError("Prepare the cache before training.")
        cache_marker = json.loads(marker.read_text(encoding="utf-8"))
        if cache_marker.get("signature") != self._cache_signature(config):
            raise ValueError("Dataset or cache settings changed. Prepare the cache again.")

        run_fields = ("rank", "lr", "seed", "resolution", "max_seq", "save_every", "attention_only", "grad_ckpt",
                      "trigger", "caption", "use_txt", "push", "private", "hub_id", "drive",
                      "preview_enabled", "preview_slots", "preview_resolution", "preview_cleanup_cache")
        run_settings = {key: config[key] for key in run_fields}
        run_manifest = {}
        if manifest_path.is_file():
            try:
                run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                run_manifest = {}
        if resume and run_manifest.get("config"):
            saved_config = run_manifest["config"]
            mismatches = [key for key in run_fields if key in saved_config and saved_config.get(key) != run_settings[key]]
            if config["preview_enabled"] and not saved_config.get("preview_enabled", False):
                mismatches.append("preview_enabled")
            if mismatches:
                raise ValueError("Resume settings differ from the saved run: " + ", ".join(sorted(set(mismatches))) +
                                 ". Keep the original settings or start a new project.")
        elif resume:
            self._append_log("Legacy run has no settings manifest; checking available checkpoint files.")

        checkpoints = []
        for candidate in output.glob("checkpoint-*"):
            try:
                step = int(candidate.name.split("-")[-1])
            except ValueError:
                continue
            if candidate.is_dir() and (candidate / "training_state.pt").is_file() and (candidate / "pytorch_lora_weights.safetensors").is_file():
                checkpoints.append((step, candidate))
        if resume and not checkpoints:
            raise ValueError("No complete local resume checkpoint was found. Enable Drive storage before training to keep full optimizer state across runtime resets.")
        latest_step = max(checkpoints)[0] if checkpoints else 0
        if resume and config["steps"] < latest_step:
            raise ValueError(f"The saved checkpoint is already at step {latest_step}; set Full run steps to at least that value.")

        config_file = output / "training_config.json"
        if resume and config_file.exists():
            previous = json.loads(config_file.read_text(encoding="utf-8"))
            if previous.get("rank") not in (None, config["rank"]):
                raise ValueError("Resume rank differs from saved run. Use Train for a new run.")

        output.mkdir(parents=True, exist_ok=True)
        preview_config = self._write_preview_config(config, root / "preview_config.json")
        run_manifest = {"schema":2, "status":"running", "project":root.name,
                        "dataset_path":str(self.dataset_dir), "cache_signature":cache_marker["signature"],
                        "config":run_settings, "target_steps":config["steps"],
                        "last_complete_step":latest_step, "updated_at":time.time()}
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)

        token = self._token()
        env = os.environ.copy()
        env["HF_TOKEN"] = token
        if resume and config["preview_enabled"] and latest_step:
            if not self._preview_complete(output, latest_step, config["preview_slots"]):
                try:
                    self._run_preview(config, cache, output, preview_config, latest_step, env)
                except GpuMemoryReleaseError:
                    raise
                except Exception as exc:
                    self._append_log(f"Turbo preview at resumed step {latest_step} failed; training will continue: {exc}")

        try:
            current_resume = bool(resume)
            while True:
                command = build_train_command(config, cache, output, current_resume)
                if config["push"]:
                    command += ["--push-to-hub", "--hub-model-id", self._repo_id(token)]
                    if config["private"]:
                        command.append("--private-repo")
                return_code = self._run_process(command, env, "train", config["steps"], allowed_return_codes=(75,))
                if return_code != 75:
                    break
                latest_checkpoint = max(
                    ((int(p.name.split("-")[-1]), p) for p in output.glob("checkpoint-*")
                     if p.is_dir() and (p / "training_state.pt").is_file()
                     and (p / "pytorch_lora_weights.safetensors").is_file()),
                    default=(0, None), key=lambda row:row[0])
                step, _checkpoint_path = latest_checkpoint
                if step <= 0:
                    raise RuntimeError("Training requested a preview but no complete resume checkpoint was found.")
                latest_step = step
                run_manifest.update({"last_complete_step":step, "updated_at":time.time()})
                manifest_tmp.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
                os.replace(manifest_tmp, manifest_path)
                try:
                    self._run_preview(config, cache, output, preview_config, step, env)
                except GpuMemoryReleaseError:
                    raise
                except Exception as exc:
                    self._append_log(f"Turbo preview at step {step} failed; saved training checkpoint is safe and training will continue: {exc}")
                current_resume = True

            final = output / "pytorch_lora_weights.safetensors"
            if not final.is_file() or final.stat().st_size < 1024:
                raise RuntimeError("Training exited without a valid final LoRA file.")
            with safe_open(str(final), framework="pt", device="cpu") as f:
                tensor_count = len(f.keys())
            if tensor_count == 0:
                raise RuntimeError("Final LoRA has no tensors.")

            if config["preview_enabled"] and not self._preview_complete(output, config["steps"], config["preview_slots"]):
                try:
                    self._run_preview(config, cache, output, preview_config, config["steps"], env)
                except Exception as exc:
                    self._append_log(f"Final Turbo preview failed; the trained LoRA is still ready: {exc}")
        except Exception:
            completed = latest_step
            for candidate in output.glob("checkpoint-*"):
                try:
                    step = int(candidate.name.split("-")[-1])
                except ValueError:
                    continue
                if (candidate / "training_state.pt").is_file() and (candidate / "pytorch_lora_weights.safetensors").is_file():
                    completed = max(completed, step)
            run_manifest.update({"status":"interrupted", "last_complete_step":completed, "updated_at":time.time()})
            manifest_tmp.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
            os.replace(manifest_tmp, manifest_path)
            raise

        run_manifest.update({"status":"complete", "last_complete_step":config["steps"], "updated_at":time.time()})
        manifest_tmp.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        self._append_log(f"Final LoRA verified: {final.name}, {tensor_count} tensors, {final.stat().st_size/2**20:.1f} MB.")
        self._append_log("Training timing · " + " · ".join(
            f"{label}: {elapsed_label(self.timings[key])}" for label, key in
            (("cold start", "train_cold_start"), ("train total", "train_total"),
             ("Turbo previews", "preview_total")) if key in self.timings) +
            (f" · {self.sec_per_step:.2f}s/step" if self.sec_per_step is not None else ""))

    def _sync(self):
        if not self.push.value:
            raise ValueError("Enable Hub upload first.")
        output = self._root() / "output"
        versions = sorted((output / "versions").glob("lora_step_*.safetensors"))
        if not versions:
            raise ValueError("No LoRA versions to sync.")
        token = self._token()
        repo_id = self._repo_id(token)
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=self.private.value, exist_ok=True)
        files = versions + [p for p in (output / "pytorch_lora_weights.safetensors",
                                             output / "training_config.json") if p.exists()]
        for index, path in enumerate(files, 1):
            repo_path = f"checkpoints/{path.name}" if path in versions else path.name
            api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path,
                            repo_id=repo_id, repo_type="model")
            self._set_progress(index / len(files) * 100, f"Hub upload · {index}/{len(files)}")
        remote = {f.rfilename for f in api.repo_info(repo_id=repo_id, repo_type="model").siblings}
        if not all((f"checkpoints/{p.name}" if p in versions else p.name) in remote for p in files):
            raise RuntimeError("Hub verification did not find every uploaded file.")
        self._append_log(f"Hub verified: https://huggingface.co/{repo_id}")

    def list_loras(self):
        """Return only complete, non-empty adapter files in download-friendly order."""
        output = self._root() / "output"
        versions = []
        for path in (output / "versions").glob("lora_step_*.safetensors"):
            match = re.fullmatch(r"lora_step_(\d+)\.safetensors", path.name)
            if not match or not path.is_file():
                continue
            size = path.stat().st_size
            if size <= 0:
                continue
            step = int(match.group(1))
            versions.append({"filename":path.name, "kind":"checkpoint", "step":step,
                             "label":f"Checkpoint · {step:,} steps", "size_bytes":size,
                             "size_label":self._format_file_size(size)})
        versions.sort(key=lambda item: item["step"], reverse=True)

        final = output / "pytorch_lora_weights.safetensors"
        final_item = None
        if final.is_file() and final.stat().st_size > 0:
            step = None
            config_path = output / "training_config.json"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                step = int(config.get("steps")) if config.get("steps") is not None else None
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            if step is None and versions:
                step = versions[0]["step"]
            size = final.stat().st_size
            label = f"Final · {step:,} steps" if step is not None else "Final LoRA"
            final_item = {"filename":final.name, "kind":"final", "step":step, "label":label,
                          "size_bytes":size, "size_label":self._format_file_size(size)}
        return ([final_item] if final_item else []) + versions

    @staticmethod
    def _format_file_size(size):
        return f"{size / 2**30:.2f} GB" if size >= 2**30 else f"{size / 2**20:.1f} MB"

    def resolve_lora_file(self, filename):
        """Resolve a listed adapter by its exact basename; reject arbitrary paths."""
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("Choose a LoRA file from the saved-files list.")
        match = next((item for item in self.list_loras() if item["filename"] == filename), None)
        if match is None:
            raise FileNotFoundError("That LoRA is no longer available. Refresh the list and try again.")
        path = (self._root() / "output" / "versions" / filename if match["kind"] == "checkpoint"
                else self._root() / "output" / filename)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError("The selected LoRA is missing or incomplete. Refresh the list and try again.")
        return path

    def refresh_results(self):
        try:
            output = self._root() / "output"
            items = self.list_loras()
            self._lora_inventory = items
            count = len(items)
            final_ready = any(item["kind"] == "final" for item in items)
            self.results.value = ("<div style='color:#000'>"
                f"<b>Results</b> · {count} LoRA file(s) · Final LoRA: {'ready' if final_ready else 'pending'}"
                f"<br><small>{html.escape(str(output))}</small></div>")
        except Exception as exc:
            self._lora_inventory = []
            self.results.value = f"Results unavailable: {html.escape(str(exc))}"

    def timing_html(self):
        live = dict(self.timings)
        now = time.monotonic()
        live_eta = self.eta_seconds
        if self.busy and self.current_stage in ("train", "resume") and self.stage_started is not None:
            if self.train_first is None:
                live["train_cold_start"] = now - self.stage_started
            else:
                live["train_total"] = now - self.stage_started
            if self.eta_seconds is not None and self.last_step_at is not None:
                live_eta = max(0.0, self.eta_seconds - (now - self.last_step_at))
        if self.busy and self.current_stage == "cache" and self.cache_started is not None:
            live["cache_total"] = now - self.cache_started
        if self.busy and self.current_stage == "preview" and self.stage_started is not None:
            live["preview_total"] = live.get("preview_total", 0.0) + now - self.stage_started
        if self.busy and self.current_stage == "cache" and self.vae_first is not None and self.vae_started is not None and self.qwen_started is None:
            live["vae_encode"] = now - self.vae_first
        if self.busy and self.current_stage == "cache" and self.qwen_first is not None:
            live["caption_encode"] = now - self.qwen_first
        def pretty(value):
            if value is None:
                return "—"
            seconds = max(0, int(value))
            return f"{seconds}s" if seconds < 60 else f"{seconds//60}m {seconds%60:02d}s"
        values = [
            ("Setup", live.get("setup")), ("VAE load", live.get("vae_load")),
            ("VAE download", live.get("vae_download")), ("VAE + encode", live.get("vae_stage", live.get("vae_encode"))),
            ("Qwen download", live.get("qwen_download")), ("Tokenizer load", live.get("tokenizer_load")),
            ("Qwen NF4 load", live.get("qwen_model_load")), ("Qwen total", live.get("qwen_load")),
            ("Caption cache", live.get("caption_encode")), ("Cache total", live.get("cache_total")),
            ("Transformer download", live.get("transformer_download")),
            ("Transformer NF4 load", live.get("transformer_nf4_load")),
            ("Trainer setup", live.get("adapter_optimizer_setup")),
            ("Cached data to RAM", live.get("cache_ram_load")),
            ("First optimizer step", live.get("first_step_compute")),
            ("Train cold start", live.get("train_cold_start")),
            ("Train loop", live.get("train_loop_total")),
            ("Seconds / step", live.get("train_steady_sec_per_step", self.sec_per_step)),
            ("Train ETA · live", live_eta), ("Turbo prefetch", live.get("preview_prefetch")),
            ("Turbo download", live.get("preview_download")), ("Turbo NF4 load", live.get("preview_nf4_load")),
            ("Turbo VAE load", live.get("preview_vae_load")), ("Turbo pipeline", live.get("preview_pipeline_load")),
            ("Turbo LoRA attach", live.get("preview_adapter_load")),
            ("Turbo model load", live.get("preview_model_load")), ("Turbo image time", live.get("preview_images_total")),
            ("VRAM return wait", live.get("preview_vram_reclaim")),
            ("Turbo previews", live.get("preview_total")),
            ("Checkpoint save", live.get("checkpoint_local")),
            ("Checkpoint upload", live.get("checkpoint_upload")),
            ("Final export", live.get("final_export_save")), ("Final upload", live.get("final_upload")),
        ]
        cards = "".join(f"<div class='metric'><small>{html.escape(label)}</small><b>{pretty(value)}</b></div>" for label, value in values)
        return f"<div class='card'><h3>Timing</h3><div class='metric-grid'>{cards}</div></div>"

    def state(self):
        self.refresh_results()
        current_status = self.status.value
        if self.busy and self.stage_started is not None:
            elapsed = int(time.monotonic() - self.stage_started)
            current_status = current_status.replace("</div>", f"<div style='color:#777;margin-top:5px'>Current process · {elapsed//60}m {elapsed%60:02d}s elapsed</div></div>", 1)
        return {"status":current_status, "progress":self.progress.value, "system":self.resources.value,
                "timing":self.timing_html(), "logs":self.log.value, "results":self.results.value,
                "loras":getattr(self, "_lora_inventory", []),
                "dataset":self.dataset_summary.value, "previews":self.preview_state(), "busy":bool(self.busy)}

    def preview_state(self):
        manifest = self._root() / "output" / "previews" / "latest.json"
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            return value.get("slots", {})
        except (OSError, ValueError, json.JSONDecodeError):
            return {}


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Krea 2 · LoRA Studio</title><style>
:root{color-scheme:light;--ink:#171717;--muted:#777;--line:#e8e8e8;--wash:#f7f7f6;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:auto;padding:30px 32px 48px}.head{display:flex;align-items:center;justify-content:space-between;gap:18px;border-bottom:1px solid var(--line);padding:3px 0 20px;margin-bottom:18px}
h1{font-size:28px;letter-spacing:-1.2px;line-height:1.15;margin:0;font-weight:680}.sub{color:var(--muted);margin-top:6px}.tag{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:99px;padding:7px 11px;color:#555;font-size:12px;white-space:nowrap}.dot{width:7px;height:7px;border-radius:50%;background:#888}.tag[data-state="online"] .dot{background:#26805a}.tag[data-state="reconnecting"] .dot{background:#b7791f}.tag[data-state="offline"] .dot{background:#a33}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}.step,.card{border:1px solid var(--line);border-radius:12px;background:#fff;box-shadow:0 2px 12px #00000005}.step{padding:12px 15px;display:flex;align-items:center;gap:10px}.num{display:grid;place-items:center;width:25px;height:25px;border-radius:50%;background:#f3f3f2;color:#555;font-size:11px;font-weight:650}.step strong{font-size:13px;font-weight:600}
.layout{display:grid;grid-template-columns:minmax(330px,.92fr) minmax(440px,1.08fr);gap:14px;align-items:start}.layout>section:last-child{grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr);gap:14px;align-items:start}.card{padding:18px;margin-bottom:14px}h2{font-size:16px;letter-spacing:-.25px;margin:0 0 3px}h3{font-size:13px;margin:0 0 9px}.note{font-size:12px;color:var(--muted);margin:0 0 14px}.field{margin:10px 0}.field label{display:block;font-size:12px;font-weight:600;margin-bottom:5px}.field input,.field select,.field textarea{width:100%;border:1px solid #dedede;border-radius:7px;background:#fff;padding:7px 10px;color:var(--ink);font:inherit;outline:none}.field input,.field select{height:37px}.field textarea{min-height:74px;resize:vertical}.field input:focus,.field select:focus,.field textarea:focus{border-color:#777;box-shadow:0 0 0 2px #0000000c}.check{display:flex;align-items:center;gap:8px;margin:9px 0;font-size:12px;color:#333}.check input{accent-color:#171717}.subtle{padding:10px 12px;background:var(--wash);border-radius:8px;margin-top:12px}
button{font-weight:600;font-size:12px;font-family:inherit;cursor:pointer;border-radius:7px;padding:10px 12px;transition:background .15s,border .15s}.primary{border:1px solid var(--ink);background:var(--ink);color:white}.primary:hover{background:#333}.secondary{border:1px solid #dedede;background:white;color:var(--ink)}.secondary:hover{background:#f5f5f5}.buttons{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px}.buttons button{flex:1}.library-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.library-head .note{margin:4px 0 0}.library-actions{display:flex;gap:7px;flex-shrink:0}.lora-list{display:grid;gap:6px;margin-top:10px;max-height:390px;overflow:auto}.lora-row{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:9px 10px;border:1px solid var(--line);border-radius:5px;background:#fff}.lora-info{min-width:0}.lora-info strong{display:block;font-size:12px;font-weight:650}.lora-info span{display:block;color:var(--muted);font-size:11px;overflow-wrap:anywhere}.lora-row button{flex:0 0 auto;padding:8px 12px}.lora-empty{padding:14px;border:1px dashed #d8d8d8;border-radius:5px;color:var(--muted);font-size:12px;text-align:center}.library-head button{white-space:nowrap}.wide{width:100%;margin-top:8px}.disabled,button:disabled{opacity:.48;cursor:wait}.status,.meter{border:1px solid var(--line);border-radius:9px;padding:11px 12px;background:#fff;margin:9px 0}.meter{font-size:12px}.bar{height:7px;background:#eee;border-radius:9px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;background:#171717;width:0;transition:width .25s}.monitor{display:grid;grid-template-columns:1fr 1fr;gap:9px}.monitor .status,.monitor .meter{margin:0;min-width:0}.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metric{padding:9px;background:var(--wash);border-radius:7px;min-width:0}.metric small{display:block;color:#777;font-size:10px;letter-spacing:.35px}.metric b{display:block;margin-top:2px;font-size:14px;overflow-wrap:anywhere}.console{background:#171717;color:#eee;border-radius:9px;padding:13px}.console pre{white-space:pre-wrap;overflow:auto;max-height:300px;margin:8px 0 0;font:11px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}.hint{font-size:11px;color:#888}.drop{border:1px dashed #d8d8d8;border-radius:8px;padding:12px;background:#fafafa}.upload-progress{height:6px;background:#eee;border-radius:8px;margin-top:9px;overflow:hidden}.upload-progress i{display:block;height:100%;width:0;background:#111}.dataset{font-size:12px;color:#555;overflow-wrap:anywhere}.section-gap{margin-top:17px}.message{min-height:18px;color:#777;font-size:11px;margin-top:6px}.err{color:#9b1c1c}details summary{cursor:pointer;font-size:12px;font-weight:600;color:#444}details[open] summary{margin-bottom:9px}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.preview-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}.preview-card{border:1px solid var(--line);border-radius:9px;padding:10px;min-width:0}.preview-card .check{margin-top:0}.preview-frame{position:relative;display:grid;place-items:center;width:100%;aspect-ratio:4/3;background:#f5f5f4;border-radius:7px;overflow:hidden;color:#888;font-size:11px;text-align:center}.preview-frame img{width:100%;height:100%;object-fit:contain}.preview-placeholder{padding:12px}.preview-meta{min-height:18px;margin-top:6px}.preview-note{padding:10px 11px;background:var(--wash);border-radius:7px;margin:10px 0;font-size:11px;color:#666}.preview-estimate{font-size:11px;font-weight:600;margin-top:8px;color:#333}.preview-card .field{margin:8px 0}.divider{height:1px;background:var(--line);margin:14px 0}
@media(max-width:850px){.wrap{padding:22px}.layout{grid-template-columns:1fr 1fr}.layout>section:last-child{grid-column:1/-1}}@media(max-width:620px){.wrap{padding:14px}.head{align-items:flex-start;flex-direction:column;gap:11px}.steps{gap:6px}.step{padding:9px;gap:7px}.step strong{font-size:11px}.layout{display:block}.layout>section:last-child{grid-column:auto;display:block}.monitor{grid-template-columns:1fr}.row,.preview-grid{grid-template-columns:1fr}.library-head{display:block}.library-actions{margin-top:10px}.library-actions button{flex:1}.lora-row{align-items:flex-start}}
</style></head><body><main class="wrap">
<header class="head"><div><h1>Krea 2 <span style="font-weight:400">LoRA Studio</span></h1><div class="sub">A simple, staged workflow for training and saving a LoRA.</div></div><div class="tag" id="connection" data-state="connecting" role="status" aria-live="polite"><span class="dot"></span><span id="connection-label">Connecting to Colab…</span></div></header>
<div class="steps" aria-label="Workflow"><div class="step"><span class="num">1</span><strong>Configure</strong></div><div class="step"><span class="num">2</span><strong>Add dataset</strong></div><div class="step"><span class="num">3</span><strong>Prepare & train</strong></div></div>
<div class="layout"><section>
<div class="card"><h2>LoRA settings</h2><p class="note">Start with these settings or adjust the advanced options.</p>
<div class="field"><label for="project">Project name</label><input id="project" value="my_krea2_lora" autocomplete="off"></div>
<div class="field"><label for="trigger">Trigger word</label><input id="trigger" value="my_trigger_word" autocomplete="off"></div>
<div class="field"><label for="caption">Fallback caption</label><input id="caption" value="character appearance and style" autocomplete="off"></div>
<label class="check"><input id="use_txt" type="checkbox" checked>Use matching .txt captions when available</label>
<div class="subtle"><label class="check"><input id="test_mode" type="checkbox">Quick run · 20 steps, rank 8</label><div class="hint">Off by default. When enabled, Train runs 20 steps at rank 8.</div></div>
<div class="row"><div class="field"><label for="resolution">Resolution</label><select id="resolution"><option>512</option><option>640</option><option>768</option></select></div><div class="field"><label for="seed">Seed</label><input id="seed" type="number" value="42"></div></div>
<details class="section-gap"><summary>Advanced training settings</summary><div class="row"><div class="field"><label for="steps">Training steps</label><input id="steps" type="number" min="1" max="100000" value="500"></div><div class="field"><label for="rank">LoRA rank</label><select id="rank"><option>4</option><option>8</option><option selected>16</option><option>32</option></select></div></div>
<div class="row"><div class="field"><label for="lr">Learning rate</label><input id="lr" type="number" step="0.00001" value="0.0003"></div><div class="field"><label for="save_every">Save every steps</label><input id="save_every" type="number" min="1" value="50"></div></div>
<div class="row"><div class="field"><label for="max_seq">Caption token limit</label><select id="max_seq"><option>96</option><option selected>128</option><option>192</option><option>256</option></select></div><div class="field"><label for="keep">Resume versions to keep</label><input id="keep" type="number" min="1" max="20" value="2"></div></div>
<label class="check"><input id="attention_only" type="checkbox" checked>Attention-only LoRA</label><label class="check"><input id="grad_ckpt" type="checkbox" checked>Gradient checkpointing · lower VRAM use</label></details>
<details class="section-gap" id="preview-options"><summary>Optional Turbo preview images</summary>
<label class="check"><input id="preview_enabled" type="checkbox">Generate images at each saved LoRA checkpoint</label>
<p class="hint">Uses a compact, pre-quantized NF4 Krea 2 Turbo model at 8 steps. Preview resolution defaults to 512 px. Prompt embeddings are reused from Prepare cache; only the transformer, scheduler and VAE are loaded for preview.</p>
<div id="preview-settings" hidden>
<div class="row"><div class="field"><label for="preview_resolution">Preview size</label><select id="preview_resolution"><option selected>512</option><option>768</option><option>1024</option></select></div><div class="field"><label>Disk space</label><div class="hint subtle">Downloads alongside Prepare cache. If needed, only the full-precision Turbo Hugging Face cache can be replaced; dataset, cache, checkpoints and LoRA files stay.</div></div></div>
<label class="check"><input id="preview_cleanup_cache" type="checkbox" checked>Replace cached full-precision Turbo weights only if disk space is tight</label>
<div id="preview-estimate" class="preview-estimate">Configure preview cards to see the image count.</div>
<div class="preview-note">Each enabled card keeps its own prompt and seed. Static seeds repeat the same sampling setup; random seeds change at each checkpoint. Training resumes from a full checkpoint even if preview generation fails.</div>
<div class="preview-grid">
<article class="preview-card"><label class="check"><input id="preview_1_enabled" type="checkbox" checked>Preview 1</label><div class="preview-frame"><img id="preview-image-1" alt="Turbo preview 1" hidden><div id="preview-placeholder-1" class="preview-placeholder">Your first preview will appear here.</div></div><div id="preview-meta-1" class="hint preview-meta">No image generated yet.</div><div class="field"><label for="preview_1_prompt">Prompt</label><textarea id="preview_1_prompt" maxlength="2000" placeholder="Describe the image to sample"></textarea></div><div class="field"><label for="preview_1_seed">Seed</label><input id="preview_1_seed" type="number" min="0" max="4294967295" value="42"></div><label class="check"><input id="preview_1_random" type="checkbox">Random seed each checkpoint</label></article>
<article class="preview-card"><label class="check"><input id="preview_2_enabled" type="checkbox">Preview 2</label><div class="preview-frame"><img id="preview-image-2" alt="Turbo preview 2" hidden><div id="preview-placeholder-2" class="preview-placeholder">Your second preview will appear here.</div></div><div id="preview-meta-2" class="hint preview-meta">No image generated yet.</div><div class="field"><label for="preview_2_prompt">Prompt</label><textarea id="preview_2_prompt" maxlength="2000" placeholder="Describe the image to sample"></textarea></div><div class="field"><label for="preview_2_seed">Seed</label><input id="preview_2_seed" type="number" min="0" max="4294967295" value="43"></div><label class="check"><input id="preview_2_random" type="checkbox">Random seed each checkpoint</label></article>
<article class="preview-card"><label class="check"><input id="preview_3_enabled" type="checkbox">Preview 3</label><div class="preview-frame"><img id="preview-image-3" alt="Turbo preview 3" hidden><div id="preview-placeholder-3" class="preview-placeholder">Your third preview will appear here.</div></div><div id="preview-meta-3" class="hint preview-meta">No image generated yet.</div><div class="field"><label for="preview_3_prompt">Prompt</label><textarea id="preview_3_prompt" maxlength="2000" placeholder="Describe the image to sample"></textarea></div><div class="field"><label for="preview_3_seed">Seed</label><input id="preview_3_seed" type="number" min="0" max="4294967295" value="44"></div><label class="check"><input id="preview_3_random" type="checkbox">Random seed each checkpoint</label></article>
<article class="preview-card"><label class="check"><input id="preview_4_enabled" type="checkbox">Preview 4</label><div class="preview-frame"><img id="preview-image-4" alt="Turbo preview 4" hidden><div id="preview-placeholder-4" class="preview-placeholder">Your fourth preview will appear here.</div></div><div id="preview-meta-4" class="hint preview-meta">No image generated yet.</div><div class="field"><label for="preview_4_prompt">Prompt</label><textarea id="preview_4_prompt" maxlength="2000" placeholder="Describe the image to sample"></textarea></div><div class="field"><label for="preview_4_seed">Seed</label><input id="preview_4_seed" type="number" min="0" max="4294967295" value="45"></div><label class="check"><input id="preview_4_random" type="checkbox">Random seed each checkpoint</label></article>
</div></div></details>
<details class="section-gap"><summary>Optional Hugging Face upload</summary><label class="check"><input id="push" type="checkbox">Upload saved LoRA versions to Hugging Face</label><label class="check"><input id="private" type="checkbox" checked>Keep repository private</label><div class="field"><label for="hub_id">Repository ID</label><input id="hub_id" placeholder="username/model-name"></div><p class="hint">Add an HF_TOKEN secret in Colab and enable notebook access before starting. Leave upload off to train and download locally.</p></details>
<details class="section-gap"><summary>Google Drive storage</summary><label class="check"><input id="drive" type="checkbox">Keep dataset, cache & resume state on Drive</label><p class="hint">Enable before adding the dataset. Drive can preserve reusable cache and resume files after a runtime reset.</p></details>
</div></section>
<section><div class="card"><h2>Dataset</h2><p class="note">Upload one ZIP with images and optional matching .txt captions.</p><div class="drop"><input id="zip" type="file" accept=".zip,application/zip"><div class="upload-progress"><i id="uploadbar"></i></div><div id="uploadlabel" class="hint">A ZIP can be uploaded in retryable chunks.</div></div><button id="uploadbtn" class="secondary wide" onclick="uploadZip()">Upload dataset</button>
<div class="divider"></div><div class="field"><label for="dataset_path">Or choose a folder already in Colab or Drive</label><input id="dataset_path" placeholder="/content/my_dataset"></div><button class="secondary wide" onclick="usePath()">Use dataset folder</button><div id="dataset" class="dataset section-gap">No dataset selected yet.</div></div>
<div class="card"><h2>Training</h2><p class="note">Each model stage exits before the next large model loads.</p><div class="hint">1 · Build reusable VAE and caption cache<br>2 · Load Krea 2 and train the adapter<br>3 · Save versions and final LoRA</div>
<button class="primary wide job" onclick="runAction('cache')">Prepare cache</button><div class="buttons"><button class="primary job" onclick="runAction('train')">Train</button><button class="secondary job" onclick="runAction('resume')">Resume</button></div>
<button class="secondary wide job" onclick="runAction('sync')">Upload saved versions</button><div id="action-message" class="message" role="status" aria-live="polite"></div></div>
<div class="card" id="lora-library"><div class="library-head"><div><h2>Saved LoRA files</h2><p class="note">Download a checkpoint or the final model. Transfers start directly and can run while training continues.</p></div><div class="library-actions"><button class="secondary" id="lora-refresh" type="button">Refresh</button><button class="primary" id="lora-download-all" type="button" disabled>Download all</button></div></div><div id="lora-summary" class="hint">No saved LoRA yet.</div><div id="lora-list" class="lora-list" aria-live="polite"><div class="lora-empty">Saved LoRA files will appear here.</div></div><div id="lora-message" class="message" role="status" aria-live="polite"></div></div></section>
<section><div class="card"><h2>Run monitor</h2><p class="note">Progress, live system readings and stage timings.</p><div class="monitor"><div id="status" class="status">Starting…</div><div id="progress" class="meter">Waiting for dataset<div class="bar"><i></i></div></div><div id="system" class="meter">Reading system metrics…</div><div id="timing" class="meter">Stage timings will appear here.</div></div></div><div class="console"><strong>Activity</strong><pre id="logs">Stage output will appear here.</pre></div></section></div></main>
<script>
const KEY=decodeURIComponent(location.hash.slice(1));let busy=false;const $=id=>document.getElementById(id);
function config(){const c={};['project','trigger','caption','resolution','preview_resolution','steps','rank','lr','save_every','max_seq','seed','keep','hub_id'].forEach(k=>c[k]=$(k).value);['use_txt','test_mode','attention_only','grad_ckpt','push','private','drive','preview_enabled','preview_cleanup_cache'].forEach(k=>c[k]=$(k).checked);for(let i=1;i<=4;i++){c['preview_'+i+'_prompt']=$('preview_'+i+'_prompt').value;c['preview_'+i+'_seed']=$('preview_'+i+'_seed').value;c['preview_'+i+'_enabled']=$('preview_'+i+'_enabled').checked;c['preview_'+i+'_random']=$('preview_'+i+'_random').checked}return c}
function updatePreviewEstimate(){const c=config(),active=[];for(let i=1;i<=4;i++)if(c['preview_'+i+'_enabled'])active.push(i);const steps=c.test_mode?Math.min(Number(c.steps)||20,20):(Number(c.steps)||500),save=Math.max(1,Number(c.save_every)||50),intervals=Math.ceil(steps/save);$('preview-estimate').textContent=!c.preview_enabled?'Preview is off · training speed is unchanged.':active.length?`${active.length} card(s) × ${intervals} checkpoints = ${active.length*intervals} planned image(s) · ${intervals} Turbo handoff(s).`:'Enable at least one card to generate previews.'}
function updatePreviewControls(){ $('preview-settings').hidden=!$('preview_enabled').checked;updatePreviewEstimate() }
const previewSeen={},previewUrls={};async function updatePreviewCards(states){for(let i=1;i<=4;i++){const state=states[String(i)],meta=$('preview-meta-'+i),img=$('preview-image-'+i),placeholder=$('preview-placeholder-'+i),enabled=$('preview_enabled').checked&&$('preview_'+i+'_enabled').checked;if(!state){meta.textContent=enabled?'Waiting for the first saved checkpoint.':'No preview generated yet.';continue}if(state.status==='loading'||state.status==='generating'||state.status==='saving'){const label={loading:'Loading Turbo',generating:'Generating',saving:'Saving image'}[state.status];meta.textContent=`${label} · step ${state.step||'—'}`;continue}if(state.status==='error'){meta.textContent='Preview failed · '+(state.error||'see Activity log');continue}if(state.status==='ready'){meta.textContent=`Step ${state.step} · seed ${state.seed} · ${Number(state.seconds||0).toFixed(1)}s${enabled?'':' · card off'}`;const key=String(state.updated_at)+'|'+state.step+'|'+state.seed;if(previewSeen[i]!==key){previewSeen[i]=key;try{const response=await fetch('/api/preview?slot='+i+'&v='+encodeURIComponent(key),{headers:{'X-Krea-Key':KEY},cache:'no-store'});if(!response.ok)throw new Error('Image fetch · HTTP '+response.status);const url=URL.createObjectURL(await response.blob()),old=previewUrls[i];img.onload=()=>{if(old)URL.revokeObjectURL(old)};img.src=url;img.hidden=false;placeholder.hidden=true;previewUrls[i]=url}catch(e){meta.textContent='Saved image could not load · '+e.message}}}}}
function setConnection(state,label){$('connection').dataset.state=state;$('connection-label').textContent=label}
async function api(path,body){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000),opts={method:body?'POST':'GET',headers:{'X-Krea-Key':KEY},signal:controller.signal};if(body){opts.headers['Content-Type']='application/json';opts.body=JSON.stringify(body)}try{const r=await fetch(path,opts);if(!r.ok){const e=new Error((await r.text()).slice(0,250)||('HTTP '+r.status));e.status=r.status;throw e}return await r.json()}finally{clearTimeout(timer)}}
function applyState(s){$('status').innerHTML=s.status;$('progress').innerHTML=s.progress+'<div class="bar"><i style="width:'+((s.progress.match(/(\d+)%/)||[])[1]||0)+'%"></i></div>';$('system').innerHTML=s.system;$('timing').innerHTML=s.timing;const logBox=$('logs'),wasBottom=logBox.scrollHeight-logBox.scrollTop-logBox.clientHeight<36,doc=new DOMParser().parseFromString(s.logs,'text/html');logBox.textContent=doc.body.textContent;if(wasBottom)logBox.scrollTop=logBox.scrollHeight;$('lora-summary').innerHTML=s.results;renderLoraFiles(s.loras||[]);$('dataset').innerHTML=s.dataset||'No dataset selected yet.';updatePreviewCards(s.previews||{});busy=s.busy;document.querySelectorAll('.job').forEach(b=>b.disabled=busy)}
let pollBusy=false,pollDelay=1000,pollTimer=null;async function poll(){if(pollBusy)return;pollBusy=true;try{applyState(await api('/api/state'));pollDelay=1000;setConnection('online','Colab connected · '+new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}))}catch(e){const expired=e.status===401||e.status===403;setConnection(expired?'offline':'reconnecting',expired?'Session link expired · reopen from Colab':'Reconnecting in '+Math.ceil(pollDelay/1000)+'s');pollDelay=Math.min(10000,pollDelay*2)}finally{pollBusy=false;clearTimeout(pollTimer);pollTimer=setTimeout(poll,pollDelay)}}
async function runAction(action){$('action-message').textContent='';try{await api('/api/action',{action:action,config:config()});await poll()}catch(e){$('action-message').textContent=e.message;$('action-message').classList.add('err')}}
async function usePath(){$('action-message').textContent='';try{await api('/api/dataset/path',{path:$('dataset_path').value,config:config()});await poll()}catch(e){$('action-message').textContent=e.message;$('action-message').classList.add('err')}}
async function uploadZip(){
 const f=$('zip').files[0];if(!f){$('uploadlabel').textContent='Choose a ZIP first.';return}if(!f.size){$('uploadlabel').textContent='The selected file is empty.';return}
 const total=f.size,chunkSize=8*1024*1024,id=(crypto.randomUUID?crypto.randomUUID():Date.now()+'_'+Math.random().toString(36).slice(2));
 async function sendChunk(start,end){
  for(let attempt=0;attempt<4;attempt++){
   try{return await new Promise((resolve,reject)=>{
    const x=new XMLHttpRequest();x.open('POST','/api/dataset/upload');x.timeout=300000;x.setRequestHeader('X-Krea-Key',KEY);x.setRequestHeader('X-Krea-Config',encodeURIComponent(JSON.stringify(config())));x.setRequestHeader('X-Filename',encodeURIComponent(f.name));x.setRequestHeader('X-Upload-Id',id);x.setRequestHeader('X-Upload-Offset',String(start));x.setRequestHeader('X-Upload-Total',String(total));
    x.upload.onprogress=e=>{if(e.lengthComputable){const done=start+e.loaded,p=Math.round(100*done/total);$('uploadbar').style.width=p+'%';$('uploadlabel').textContent='Uploading · '+(done/1048576).toFixed(1)+' / '+(total/1048576).toFixed(1)+' MB'}};
    x.onload=()=>{let v={};try{v=JSON.parse(x.responseText)}catch(e){}if(x.status>=300){const error=new Error(v.error||'Upload failed · HTTP '+x.status);error.retryable=x.status>=500;reject(error)}else resolve(v)};
    x.onerror=()=>reject(new Error('Upload connection failed.'));x.ontimeout=()=>reject(new Error('Upload timed out.'));x.send(f.slice(start,end))
   })}catch(error){
    if(attempt===3||error.retryable===false)throw error;
    const seconds=2**attempt;$('uploadlabel').textContent='Connection interrupted · retrying this chunk in '+seconds+'s…';
    await new Promise(resolve=>setTimeout(resolve,seconds*1000))
   }
  }
 }
 try{for(let start=0;start<total;start+=chunkSize){await sendChunk(start,Math.min(total,start+chunkSize))}$('uploadbar').style.width='100%';$('uploadlabel').textContent='ZIP received · validating images…';await poll()}catch(e){$('uploadlabel').textContent=e.message}
}
let loraSignature="";
function escapeHtml(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function renderLoraFiles(items){const signature=JSON.stringify(items);if(signature===loraSignature)return;loraSignature=signature;const list=$('lora-list');$('lora-download-all').disabled=!items.length;if(!items.length){list.innerHTML='<div class="lora-empty">No saved LoRA yet. Files will appear here after a checkpoint is saved.</div>';return}list.innerHTML=items.map(item=>`<div class="lora-row"><div class="lora-info"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.filename)} · ${escapeHtml(item.size_label)}</span></div><button class="secondary" type="button" data-lora-file="${escapeHtml(item.filename)}">Download</button></div>`).join('')}
async function refreshLoraList(){const button=$('lora-refresh'),message=$('lora-message');button.disabled=true;message.textContent='Refreshing saved files…';try{const data=await api('/api/loras');renderLoraFiles(data.files||[]);$('lora-summary').innerHTML=data.summary||$('lora-summary').innerHTML;message.textContent=`List updated · ${(data.files||[]).length} file(s)`}catch(e){message.textContent=e.message;message.classList.add('err')}finally{button.disabled=false}}
async function startLoraDownload(payload){const message=$('lora-message');message.classList.remove('err');message.textContent=payload.kind==='all'?'Preparing ZIP…':'Preparing download…';try{const data=await api('/api/download-ticket',payload),a=document.createElement('a');a.href='/api/download?ticket='+encodeURIComponent(data.ticket);a.download=data.filename;document.body.appendChild(a);a.click();a.remove();message.textContent='Download started · '+data.filename}catch(e){message.textContent=e.message;message.classList.add('err')}}
function downloadAllLoras(){startLoraDownload({kind:'all'})}
$('lora-list').addEventListener('click',event=>{const button=event.target.closest('[data-lora-file]');if(button)startLoraDownload({kind:'file',filename:button.dataset.loraFile})});
$('lora-refresh').addEventListener('click',refreshLoraList);$('lora-download-all').addEventListener('click',downloadAllLoras);
let metricBusy=false;async function refreshMetrics(){if(metricBusy)return;metricBusy=true;try{const s=await api('/api/metrics');$('system').innerHTML=s.html}catch(e){}finally{metricBusy=false;setTimeout(refreshMetrics,2000)}}
document.addEventListener('visibilitychange',()=>{if(!document.hidden){clearTimeout(pollTimer);poll();refreshMetrics();refreshLoraList()}});
$('preview_enabled').addEventListener('change',updatePreviewControls);for(let i=1;i<=4;i++){['enabled','prompt','seed','random'].forEach(k=>$('preview_'+i+'_'+k).addEventListener('input',updatePreviewEstimate));$('preview_'+i+'_enabled').addEventListener('change',updatePreviewEstimate)}
if(KEY){updatePreviewControls();poll();refreshMetrics()}else{updatePreviewControls();setConnection('offline','Open the temporary Studio link from Colab');$('status').textContent='Open the private Studio link printed by the notebook.'}
</script></body></html>'''


def run_web_ui(studio):
    import secrets
    import urllib.parse
    api_key = secrets.token_urlsafe(24)
    job_lock = threading.Lock()
    job_running = {"value": False}
    upload_lock = threading.Lock()
    uploads = {}
    download_tickets = {}
    download_ticket_lock = threading.Lock()
    download_archive_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_bytes(self, code, body, content_type="application/json"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code, value):
            self.send_bytes(code, json.dumps(value, ensure_ascii=False), "application/json; charset=utf-8")

        def authorized(self):
            import hmac
            return hmac.compare_digest(self.headers.get("X-Krea-Key", ""), api_key)

        def read_json(self):
            size = int(self.headers.get("Content-Length", "0"))
            if size > 1_000_000:
                raise ValueError("Request is too large.")
            return json.loads(self.rfile.read(size) or b"{}")

        def send_lora_file(self, target):
            size = target.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                with target.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def apply_config(self, values):
            mapping = {"project":"project", "trigger":"trigger", "caption":"caption", "use_txt":"use_txt",
                "test_mode":"test_mode", "resolution":"resolution", "steps":"steps", "rank":"rank",
                "lr":"lr", "save_every":"save_every", "max_seq":"max_seq", "attention_only":"attention_only",
                "grad_ckpt":"grad_ckpt", "seed":"seed", "push":"push", "hub_id":"hub_id",
                "private":"private", "keep":"keep", "drive":"drive", "preview_enabled":"preview_enabled",
                "preview_resolution":"preview_resolution", "preview_cleanup_cache":"preview_cleanup_cache"}
            for slot in range(1, 5):
                for suffix in ("enabled", "prompt", "seed", "random"):
                    key = f"preview_{slot}_{suffix}"
                    mapping[key] = key
            for key, attr in mapping.items():
                if key in values:
                    current = getattr(studio, attr).value
                    value = values[key]
                    if isinstance(current, bool):
                        value = bool(value)
                    elif isinstance(current, int):
                        value = int(value)
                    elif isinstance(current, float):
                        value = float(value)
                    else:
                        value = str(value)
                    getattr(studio, attr).value = value

        def do_GET(self):
            route = urlparse(self.path).path
            query = parse_qs(urlparse(self.path).query)
            if route == "/":
                self.send_bytes(200, PAGE, "text/html; charset=utf-8")
            elif route == "/api/download" and "ticket" in query:
                ticket_values = query.get("ticket", [])
                if len(ticket_values) != 1:
                    self.send_json(404, {"error":"Download link is invalid or expired."})
                    return
                with download_ticket_lock:
                    entry = download_tickets.pop(ticket_values[0], None)
                if not entry or entry["expires_at"] < time.time():
                    self.send_json(404, {"error":"Download link is invalid, expired, or already used. Click Download again."})
                    return
                target = Path(entry["path"])
                try:
                    listed = studio.resolve_lora_file(entry["filename"]) if entry.get("kind") == "file" else target
                    if listed.resolve() != target.resolve() or not target.is_file() or target.stat().st_size <= 0:
                        raise FileNotFoundError
                    self.send_lora_file(target)
                except (OSError, ValueError, FileNotFoundError):
                    self.send_json(404, {"error":"This LoRA is no longer available. Refresh the list and try again."})
            elif not self.authorized():
                self.send_json(401, {"error":"Open the private session link printed by the notebook."})
            elif route == "/api/state":
                self.send_json(200, studio.state())
            elif route == "/api/loras":
                studio.refresh_results()
                self.send_json(200, {"files":studio.list_loras(), "summary":studio.results.value})
            elif route == "/api/preview":
                slot_values = query.get("slot", [])
                if len(slot_values) != 1 or slot_values[0] not in {"1", "2", "3", "4"}:
                    self.send_json(400, {"error":"Preview slot must be between 1 and 4."})
                    return
                image = studio._root() / "output" / "previews" / f"slot_{slot_values[0]}" / "latest.png"
                if not image.is_file():
                    self.send_json(404, {"error":"This preview card has no generated image yet."})
                    return
                self.send_bytes(200, image.read_bytes(), "image/png")
            elif route == "/api/metrics":
                studio._render_resources()
                self.send_json(200, {"html":studio.resources.value})
            else:
                self.send_json(404, {"error":"Not found"})

        def do_POST(self):
            if not self.authorized():
                self.send_json(401, {"error":"Session link is missing or expired."})
                return
            route = urlparse(self.path).path
            try:
                if route == "/api/download-ticket":
                    payload = self.read_json()
                    kind = payload.get("kind", "file")
                    output = studio._root() / "output"
                    if kind == "all":
                        with download_archive_lock:
                            studio.refresh_results()
                            items = studio.list_loras()
                            if not items:
                                raise ValueError("No saved LoRA files yet.")
                            target = output / f"{studio._root().name}_lora_versions.zip"
                            pending = target.with_name("." + target.name + "." + secrets.token_hex(6) + ".pending")
                            try:
                                with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED) as archive:
                                    for item in items:
                                        path = studio.resolve_lora_file(item["filename"])
                                        archive.write(path, arcname=path.name)
                                os.replace(pending, target)
                            finally:
                                pending.unlink(missing_ok=True)
                        filename = target.name
                        ticket_kind = "archive"
                    elif kind == "file":
                        filename = payload.get("filename", "")
                        target = studio.resolve_lora_file(filename)
                        ticket_kind = "file"
                    else:
                        raise ValueError("Choose one saved LoRA or Download all.")
                    if not target.is_file() or target.stat().st_size <= 0:
                        raise FileNotFoundError("The selected download is missing or empty.")
                    ticket = secrets.token_urlsafe(24)
                    with download_ticket_lock:
                        now = time.time()
                        for expired in [key for key, value in download_tickets.items() if value["expires_at"] < now]:
                            download_tickets.pop(expired, None)
                        download_tickets[ticket] = {"path":str(target.resolve()), "filename":filename,
                                                    "kind":ticket_kind, "expires_at":now + 300}
                    self.send_json(200, {"ticket":ticket, "filename":filename, "expires_in":300})
                elif route == "/api/dataset/upload":
                    cfg = json.loads(urllib.parse.unquote(self.headers.get("X-Krea-Config", "%7B%7D")))
                    self.apply_config(cfg)
                    size = int(self.headers.get("Content-Length", "0"))
                    total = int(self.headers.get("X-Upload-Total", "0"))
                    offset = int(self.headers.get("X-Upload-Offset", "0"))
                    if size < 1 or size > 40 * 2**20 or total < size or total > 5 * 2**30:
                        raise ValueError("Upload chunks must be under 40 MB and ZIPs under 5 GB.")
                    filename = Path(urllib.parse.unquote(self.headers.get("X-Filename", "dataset.zip"))).name
                    if Path(filename).suffix.lower() != ".zip":
                        raise ValueError("Choose a .zip file.")
                    upload_id = self.headers.get("X-Upload-Id", "")
                    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", upload_id):
                        raise ValueError("Invalid upload session.")
                    with upload_lock:
                        entry = uploads.get(upload_id)
                        if offset == 0 and entry is None:
                            entry = {"path":Path("/content") / ("krea_upload_" + secrets.token_hex(8) + ".zip"),
                                     "total":total, "received":0, "chunks":{}, "processed":False}
                            uploads[upload_id] = entry
                        if entry is None or entry["total"] != total:
                            raise ValueError("Upload session expired. Start the upload again.")
                        if not hasattr(entry["path"], "parent"):
                            raise ValueError("Upload session is invalid.")
                        if offset != entry["received"] and not (offset < entry["received"] and offset in entry["chunks"]):
                            raise ValueError("Upload chunk is out of order. Start the upload again.")
                        digest = hashlib.sha256()
                        append_chunk = offset == entry["received"]
                        mode = ("r+b" if entry["path"].exists() else "wb") if append_chunk else "rb"
                        with entry["path"].open(mode) as stream:
                            stream.seek(offset)
                            remaining = size
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("Upload ended early.")
                                digest.update(chunk)
                                if append_chunk:
                                    stream.write(chunk)
                                remaining -= len(chunk)
                        chunk_hash = digest.hexdigest()
                        if append_chunk:
                            entry["chunks"][offset] = {"size":size, "sha256":chunk_hash}
                            entry["received"] += size
                        elif entry["chunks"].get(offset) != {"size":size, "sha256":chunk_hash}:
                            raise ValueError("Retried upload chunk did not match the accepted data.")
                        received = entry["received"]
                        uploaded = entry["path"]
                        complete = received == total
                        processed = entry.get("processed", False)
                    if not complete:
                        self.send_json(202, {"ok":True, "received":received, "total":total})
                        return
                    if not processed:
                        studio.dataset_path.value = str(uploaded)
                        studio._use_path()
                        with upload_lock:
                            uploads[upload_id]["processed"] = True
                    self.send_json(200, {"ok":True, "received":received, "total":total})
                elif route == "/api/dataset/path":
                    body = self.read_json()
                    self.apply_config(body.get("config", {}))
                    studio.dataset_path.value = body.get("path", "")
                    studio._use_path()
                    self.send_json(200, {"ok":True})
                elif route == "/api/action":
                    body = self.read_json()
                    self.apply_config(body.get("config", {}))
                    action = body.get("action", "")
                    actions = {"cache":studio._prepare_cache, "train":lambda:studio._train(False),
                               "resume":lambda:studio._train(True), "sync":studio._sync}
                    if action not in actions:
                        raise ValueError("Unknown training action.")
                    with job_lock:
                        if job_running["value"]:
                            raise ValueError("A training stage is already running.")
                        job_running["value"] = True
                    def worker():
                        try:
                            studio._start(action, actions[action])
                        finally:
                            with job_lock:
                                job_running["value"] = False
                    threading.Thread(target=worker, daemon=True, name="krea2-stage").start()
                    self.send_json(202, {"ok":True})
                else:
                    self.send_json(404, {"error":"Not found"})
            except Exception as exc:
                self.send_json(400, {"error":str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", 7860), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="krea2-web").start()
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return "http://127.0.0.1:7860/#" + api_key, server, None

    binary = shutil.which("cloudflared")
    if not binary:
        binary = "/content/cloudflared"
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
        os.chmod(binary, 0o755)
    tunnel = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7860"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    import selectors
    selector = selectors.DefaultSelector()
    selector.register(tunnel.stdout, selectors.EVENT_READ)
    pending, public_url, deadline = "", None, time.monotonic() + 90
    while time.monotonic() < deadline and tunnel.poll() is None and not public_url:
        if selector.select(timeout=1):
            chunk = os.read(tunnel.stdout.fileno(), 4096).decode("utf-8", errors="replace")
            pending += chunk
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", pending, re.I)
            if match:
                public_url = match.group(0)
    selector.close()
    if not public_url:
        tunnel.terminate()
        server.shutdown()
        raise RuntimeError("Cloudflare quick tunnel did not return a URL within 90 seconds.")
    return public_url + "/#" + api_key, server, tunnel


try:
    KREA_STUDIO.close()
except NameError:
    pass
KREA_STUDIO = KreaStudio()
try:
    KREA_SERVER.shutdown()
    KREA_SERVER.server_close()
except Exception:
    pass
try:
    KREA_TUNNEL.terminate()
except Exception:
    pass
KREA_URL, KREA_SERVER, KREA_TUNNEL = run_web_ui(KREA_STUDIO)
display(HTML(f"<div style='font:16px Arial;padding:20px;border:1px solid #ddd;border-radius:10px'>"
    f"<b>Krea 2 LoRA Studio is ready.</b><br><br>Open the control panel in a new tab: "
    f"<a href='{html.escape(KREA_URL, quote=True)}' target='_blank' rel='noopener'>Open Krea 2 Studio</a>"
    "<br><small>The temporary session link stays active while this Colab runtime is running.</small></div>"))
