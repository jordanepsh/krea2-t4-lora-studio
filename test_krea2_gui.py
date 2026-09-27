"""CPU-only checks for the portable notebook and its dataset/process flow."""

import ast
import io
import json
import os
from types import ModuleType
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


HERE = Path(__file__).parent
NOTEBOOK = HERE / "Krea2_T4_LoRA_Studio.ipynb"
GUI = HERE / "krea2_gui_cell.py"
DATASET = HERE / "Sili_IllustriousXL2_TRAIN_READY_KreaCaptions.zip"


def load_class():
    tree = ast.parse(GUI.read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KreaStudio")
    field = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Field")
    duration = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "elapsed_label")
    command_builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_train_command")
    preview_command_builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_preview_command")
    code = compile(ast.Module(body=[field, duration, command_builder, preview_command_builder, klass], type_ignores=[]), str(GUI), "exec")
    import hashlib, html, os, shutil, subprocess, threading, zipfile
    ns = dict(Path=Path, Image=Image, hashlib=hashlib, html=html, json=json, os=os,
              re=re, shutil=shutil, subprocess=subprocess, sys=sys,
              threading=threading, time=time, zipfile=zipfile, globals=globals,
              GpuMemoryReleaseError=type("GpuMemoryReleaseError", (RuntimeError,), {}),
              PREVIEW_MODEL_REPO="OzzyGT/Krea_2_Turbo_bnb_nf4",
              PREVIEW_MODEL_REVISION="5458debf8356a6646a5aa814de28dcea881f8a6d",
              PREVIEW_MODEL_PATTERNS=["model_index.json", "scheduler/*", "transformer/*"],
              PREVIEW_MODEL_MIN_BYTES=8 * 1024**3, PREVIEW_DISK_RESERVE_BYTES=2 * 1024**3)
    validator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_preview_snapshot")
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(GUI), "exec"), ns)
    exec(code, ns)
    return ns["KreaStudio"]


class KreaGuiTests(unittest.TestCase):
    def test_shareable_defaults_and_clean_studio_copy(self):
        tree = ast.parse(GUI.read_text(encoding="utf-8"))
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KreaStudio")
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        defaults = next(ast.literal_eval(n.value) for n in ast.walk(init)
                        if isinstance(n, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "defaults" for t in n.targets))
        self.assertEqual(defaults["project"], "my_krea2_lora")
        self.assertEqual(defaults["trigger"], "my_trigger_word")
        self.assertFalse(defaults["test_mode"])
        self.assertFalse(defaults["push"])
        self.assertFalse(defaults["preview_enabled"])
        self.assertTrue(defaults["preview_1_enabled"])

        page = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "PAGE" for t in n.targets))
        self.assertIn(">Train</button>", page)
        self.assertIn("Quick run · 20 steps, rank 8", page)
        self.assertIn("AbortController()", page)
        self.assertIn("chunkSize=8*1024*1024", page)
        self.assertIn("x.timeout=300000", page)
        self.assertIn("Reconnecting in ", page)
        self.assertNotIn("Smoke test", page)
        self.assertNotIn("Train smoke test", page)
        self.assertNotIn("sili_krea2_test", page)
        self.assertNotIn("anime-style illustration of Sili", page)
        self.assertIn('id="push" type="checkbox">Upload saved LoRA versions', page)
        self.assertIn('id="connection"', page)
        self.assertIn('id="preview-image-4"', page)
        self.assertIn('id="lora-list"', page)
        self.assertIn('id="lora-refresh"', page)
        self.assertIn('id="lora-download-all"', page)
        self.assertIn("/api/download-ticket", page)
        self.assertNotIn("downloadFile('final')", page)
        self.assertNotIn("downloadFile('all')", page)
        self.assertIn("Krea 2 Turbo", page)
        self.assertIn("512 px", page)
        self.assertIn("Replace cached full-precision Turbo weights", page)
        self.assertIn("planned image(s)", page)

    def test_gradient_checkpoint_toggle_builds_expected_train_command(self):
        source = ast.parse(GUI.read_text(encoding="utf-8"))
        command_builder = next(n for n in source.body
                               if isinstance(n, ast.FunctionDef) and n.name == "build_train_command")
        ns = {}
        exec(compile(ast.Module(body=[command_builder], type_ignores=[]), str(GUI), "exec"), ns)
        config = {"steps": 20, "rank": 16, "lr": .0003, "seed": 42,
                  "save_every": 50, "keep": 2, "grad_ckpt": True, "attention_only": True}
        enabled = ns["build_train_command"](config, Path("/cache"), Path("/output"))
        self.assertIn("--grad-ckpt", enabled)
        self.assertIn("--attention-only", enabled)
        self.assertEqual(enabled[enabled.index("--save-every") + 1], "20")
        config["grad_ckpt"] = False
        disabled = ns["build_train_command"](config, Path("/cache"), Path("/output"), resume=True)
        self.assertNotIn("--grad-ckpt", disabled)
        self.assertIn("--attention-only", disabled)
        self.assertIn("--resume", disabled)
        config["preview_enabled"] = True
        preview_enabled_command = ns["build_train_command"](config, Path("/cache"), Path("/output"))
        self.assertIn("--preview-at-checkpoints", preview_enabled_command)
        config["preview_enabled"] = False
        config["attention_only"] = False
        full_targets = ns["build_train_command"](config, Path("/cache"), Path("/output"))
        self.assertNotIn("--attention-only", full_targets)

    def test_preview_config_validation_and_command(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        values = {"steps": 100, "rank": 16, "test_mode": False, "lr": .0003, "trigger": "trigger",
                  "resolution": 512, "preview_resolution":512, "preview_cleanup_cache":True,
                  "max_seq": 128, "save_every": 50, "seed": 42,
                  "attention_only": True, "grad_ckpt": True, "push": False, "private": True,
                  "keep": 2, "hub_id": "", "drive": False, "caption": "caption", "use_txt": True,
                  "preview_enabled": True}
        for slot in range(1, 5):
            values[f"preview_{slot}_enabled"] = slot in (1, 3)
            values[f"preview_{slot}_prompt"] = f"prompt {slot}" if slot in (1, 3) else ""
            values[f"preview_{slot}_seed"] = 40 + slot
            values[f"preview_{slot}_random"] = slot == 3
        for name, value in values.items():
            setattr(studio, name, SimpleNamespace(value=value))
        config = studio._config()
        self.assertEqual([slot["slot"] for slot in config["preview_slots"]], [1, 3])
        self.assertEqual(config["preview_slots"][1]["seed"], 43)
        self.assertTrue(config["preview_slots"][1]["random_seed"])

        ns = {}
        tree = ast.parse(GUI.read_text(encoding="utf-8"))
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_preview_command")
        exec(compile(ast.Module(body=[builder], type_ignores=[]), str(GUI), "exec"), ns)
        command = ns["build_preview_command"](config, Path("/cache"), Path("/v.safetensors"),
                                               Path("/out"), Path("/preview.json"), 50)
        self.assertIn("/content/krea2_t4_preview.py", command)
        self.assertEqual(command[command.index("--step") + 1], "50")
        self.assertEqual(command[command.index("--resolution") + 1], "512")

        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        backend = "".join(notebook["cells"][2]["source"])
        assignments = {n.targets[0].id: ast.literal_eval(n.value)
                       for n in ast.parse(backend).body if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == "PRECACHE_SOURCE"}
        precache_tree = ast.parse(assignments["PRECACHE_SOURCE"])
        caption_key_fn = next(n for n in precache_tree.body
                              if isinstance(n, ast.FunctionDef) and n.name == "caption_key")
        key_ns = {"hashlib":__import__("hashlib")}
        exec(compile(ast.Module(body=[caption_key_fn], type_ignores=[]), "caption_key", "exec"), key_ns)
        digest = key_ns["caption_key"]("krea/Krea-2-Raw", 128, "portrait of Sili")
        embed_path, mask_path = KreaStudio._preview_embedding_paths(Path("/cache"), "portrait of Sili", 128)
        self.assertEqual(embed_path.name, digest + "_embed.pt")
        self.assertEqual(mask_path.name, digest + "_mask.pt")

        studio.preview_1_prompt.value = ""
        with self.assertRaisesRegex(ValueError, "Preview card 1"):
            studio._config()

    def test_preview_prefetch_only_replaces_full_precision_turbo_cache_when_disk_is_tight(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        log = []
        studio._append_log = log.append
        gib = 1024**3
        deleted = []
        cleanup_execute = __import__("unittest.mock", fromlist=["Mock"]).Mock()
        old_repo = SimpleNamespace(repo_id="krea/Krea-2-Turbo", revisions=[SimpleNamespace(commit_hash="old-full-precision-rev")])

        class CacheInfo:
            repos = [old_repo]
            def delete_revisions(self, *revisions):
                deleted.extend(revisions)
                return SimpleNamespace(expected_freed_size=24 * gib, execute=cleanup_execute)

        cache_info = CacheInfo()
        sibling_rows = [
            SimpleNamespace(rfilename="model_index.json", size=500),
            SimpleNamespace(rfilename="scheduler/scheduler_config.json", size=500),
            SimpleNamespace(rfilename="transformer/config.json", size=1000),
            SimpleNamespace(rfilename="transformer/diffusion_pytorch_model.safetensors", size=8 * gib),
            SimpleNamespace(rfilename="text_encoder/model.safetensors", size=25 * gib),
            SimpleNamespace(rfilename="vae/diffusion_pytorch_model.safetensors", size=4 * gib),
        ]
        fake_hub = ModuleType("huggingface_hub")
        model_info_calls = []
        fake_hub.HfApi = lambda token=None: SimpleNamespace(
            model_info=lambda repo_id, revision, files_metadata: (
                model_info_calls.append((repo_id, revision, files_metadata)) or
                SimpleNamespace(siblings=sibling_rows)))
        fake_hub.scan_cache_dir = lambda: cache_info
        downloaded = []
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "5458debf8356a6646a5aa814de28dcea881f8a6d"
            (snapshot / "scheduler").mkdir(parents=True)
            (snapshot / "transformer").mkdir()
            for relative in ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json"):
                (snapshot / relative).write_text("{}", encoding="utf-8")
            (snapshot / "transformer/diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            def fake_snapshot_download(**kwargs):
                downloaded.append(kwargs)
                if kwargs.get("local_files_only"):
                    raise FileNotFoundError("snapshot is not cached")
                return str(snapshot)
            fake_hub.snapshot_download = fake_snapshot_download
            disk_states = iter([SimpleNamespace(free=5 * gib), SimpleNamespace(free=20 * gib), SimpleNamespace(free=20 * gib)])
            with patch.dict(sys.modules, {"huggingface_hub":fake_hub}):
                with patch("shutil.disk_usage", side_effect=lambda path: next(disk_states)):
                    studio._prefetch_preview_model("test-token", True)

        self.assertEqual(deleted, ["old-full-precision-rev"])
        cleanup_execute.assert_called_once_with()
        self.assertEqual([item.get("local_files_only", False) for item in downloaded], [True, False])
        self.assertEqual(downloaded[1]["repo_id"], "OzzyGT/Krea_2_Turbo_bnb_nf4")
        self.assertEqual(downloaded[1]["revision"], "5458debf8356a6646a5aa814de28dcea881f8a6d")
        self.assertEqual(downloaded[1]["allow_patterns"], ["model_index.json", "scheduler/*", "transformer/*"])
        self.assertEqual(model_info_calls, [("OzzyGT/Krea_2_Turbo_bnb_nf4", "5458debf8356a6646a5aa814de28dcea881f8a6d", True)])
        self.assertTrue(any("Dataset, cache, checkpoints and LoRA outputs are kept" in line for line in log))
        self.assertTrue(studio.preview_prefetch_result["ready"], studio.preview_prefetch_result)
        studio._require_preview_prefetch_ready()
        self.assertIn("preview_prefetch", studio.timings)

    def test_preview_prefetch_reuses_a_valid_local_turbo_snapshot(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        studio._append_log = lambda line: None
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.scan_cache_dir = lambda: SimpleNamespace(repos=[])
        fake_hub.HfApi = lambda token=None: (_ for _ in ()).throw(AssertionError("size API should not be needed"))
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "5458debf8356a6646a5aa814de28dcea881f8a6d"
            (snapshot / "scheduler").mkdir(parents=True)
            (snapshot / "transformer").mkdir()
            for relative in ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json"):
                (snapshot / relative).write_text("{}", encoding="utf-8")
            (snapshot / "transformer/model.safetensors").write_bytes(b"weights")
            def download(**kwargs):
                calls.append(kwargs)
                if not kwargs.get("local_files_only"):
                    raise AssertionError("valid cached snapshot should not be downloaded again")
                return str(snapshot)
            fake_hub.snapshot_download = download
            with patch.dict(sys.modules, {"huggingface_hub":fake_hub}):
                with patch("shutil.disk_usage", return_value=SimpleNamespace(free=4 * 1024**3)):
                    studio._prefetch_preview_model("test-token", False)
        self.assertTrue(studio.preview_prefetch_result["ready"], studio.preview_prefetch_result)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])
        studio._require_preview_prefetch_ready()

    def test_preview_prefetch_refuses_unverified_download_size(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        log = []
        studio._append_log = log.append
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.scan_cache_dir = lambda: SimpleNamespace(repos=[])
        fake_hub.HfApi = lambda token=None: SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="transformer/weights.safetensors", size=None, lfs=None)]))
        calls = []
        def download(**kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("snapshot missing")
            raise AssertionError("unverified size must prevent network download")
        fake_hub.snapshot_download = download
        with patch.dict(sys.modules, {"huggingface_hub":fake_hub}):
            studio._prefetch_preview_model("test-token", False)
        self.assertFalse(studio.preview_prefetch_result["ready"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])
        self.assertIn("exact sizes", studio.preview_prefetch_result["error"])
        with self.assertRaisesRegex(RuntimeError, "training was not started"):
            studio._require_preview_prefetch_ready()

    def test_failed_or_skipped_preview_prefetch_blocks_a_run_with_previews(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.preview_prefetch_result = {"ready": False, "error": "network unavailable"}
        with self.assertRaisesRegex(RuntimeError, "training was not started.*network unavailable"):
            studio._require_preview_prefetch_ready()

    def test_preview_worker_downloads_a_missing_snapshot_once_when_disk_allows(self):
        source = Path(__file__).with_name("krea2_turbo_preview.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and getattr(node, "name", "") in {"IncompletePreviewSnapshot", "validate_snapshot", "expected_snapshot_bytes", "resolve_snapshot"}]
        fake_local_error = type("LocalEntryNotFoundError", (RuntimeError,), {})
        ns = {"Path":Path, "shutil":__import__("shutil"), "LocalEntryNotFoundError":fake_local_error,
              "PREVIEW_DISK_RESERVE_BYTES":2 * 1024**3, "fnmatchcase":__import__("fnmatch").fnmatchcase}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), ns)
        calls = []
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.HfApi = lambda token=None: SimpleNamespace(model_info=lambda **kwargs: SimpleNamespace(siblings=[
            SimpleNamespace(rfilename="model_index.json", size=100),
            SimpleNamespace(rfilename="scheduler/scheduler_config.json", size=100),
            SimpleNamespace(rfilename="transformer/config.json", size=100),
            SimpleNamespace(rfilename="transformer/weights.safetensors", size=8 * 1024**3),
        ]))
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "5458debf8356a6646a5aa814de28dcea881f8a6d"
            (snapshot / "scheduler").mkdir(parents=True)
            (snapshot / "transformer").mkdir()
            for relative in ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json"):
                (snapshot / relative).write_text("{}", encoding="utf-8")
            (snapshot / "transformer/weights.safetensors").write_bytes(b"weights")
            def fake_download(**kwargs):
                calls.append(kwargs)
                if kwargs.get("local_files_only"):
                    raise fake_local_error("missing pinned revision")
                return str(snapshot)
            ns["snapshot_download"] = fake_download
            with patch.dict(sys.modules, {"huggingface_hub":fake_hub}):
                with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
                    resolved = ns["resolve_snapshot"]("OzzyGT/Krea_2_Turbo_bnb_nf4", revision=snapshot.name,
                    allow_patterns=["model_index.json", "scheduler/*", "transformer/*"], token=None,
                    component="Turbo", min_free_bytes=8 * 1024**3)
        self.assertEqual(resolved, snapshot)
        self.assertEqual([call["local_files_only"] for call in calls], [True, False])

    def test_preview_worker_refuses_to_download_when_disk_reserve_would_be_breached(self):
        source = Path(__file__).with_name("krea2_turbo_preview.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and getattr(node, "name", "") in {"IncompletePreviewSnapshot", "validate_snapshot", "expected_snapshot_bytes", "resolve_snapshot"}]
        fake_local_error = type("LocalEntryNotFoundError", (RuntimeError,), {})
        ns = {"Path":Path, "shutil":__import__("shutil"), "LocalEntryNotFoundError":fake_local_error,
              "PREVIEW_DISK_RESERVE_BYTES":2 * 1024**3, "fnmatchcase":__import__("fnmatch").fnmatchcase}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), ns)
        calls = []
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.HfApi = lambda token=None: SimpleNamespace(model_info=lambda **kwargs: SimpleNamespace(siblings=[
            SimpleNamespace(rfilename="model_index.json", size=8 * 1024**3),
        ]))
        ns["snapshot_download"] = lambda **kwargs: calls.append(kwargs) or (_ for _ in ()).throw(fake_local_error("missing"))
        with patch.dict(sys.modules, {"huggingface_hub":fake_hub}):
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=9 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, "cannot be downloaded safely"):
                    ns["resolve_snapshot"]("OzzyGT/Krea_2_Turbo_bnb_nf4", revision="pinned",
                    allow_patterns=["model_index.json"], token=None, component="Turbo",
                    min_free_bytes=8 * 1024**3)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])

    def test_preview_waits_for_vram_to_return_to_baseline(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        log = []
        studio._append_log = log.append
        readings = iter([(8000, 15360), (3900, 15360)])
        studio._gpu_memory_snapshot = lambda: next(readings)
        with patch("time.sleep", lambda seconds: None):
            studio._wait_for_gpu_release((4000, 15360), timeout=1)
        self.assertGreaterEqual(studio.timings["preview_vram_reclaim"], 0.0)
        self.assertTrue(any("pre-preview baseline" in line for line in log))

    def test_preview_refuses_to_resume_training_if_vram_stays_allocated(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        studio._append_log = lambda line: None
        studio._gpu_memory_snapshot = lambda: (6000, 15360)
        with self.assertRaisesRegex(RuntimeError, "Training remains at its saved checkpoint"):
            studio._wait_for_gpu_release((1000, 15360), timeout=0)

    def test_preview_png_is_saved_atomically_and_latest_uses_link_when_available(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = "".join(notebook["cells"][2]["source"])
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(cell).body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PREVIEW_SOURCE"
        }
        preview = assignments["PREVIEW_SOURCE"]
        function = next(node for node in ast.parse(preview).body
                        if isinstance(node, ast.FunctionDef) and node.name == "save_preview_files")
        namespace = {"Path":Path, "time":time, "os":os, "shutil":__import__("shutil")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "save_preview_files", "exec"), namespace)

        class FakeImage:
            def save(self, path, format, compress_level):
                self.settings = (format, compress_level)
                Path(path).write_bytes(b"png-data")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "slot_1"
            image = FakeImage()
            seconds = namespace["save_preview_files"](image, folder / "step_00020.png", folder / "latest.png")
            self.assertEqual((folder / "step_00020.png").read_bytes(), b"png-data")
            self.assertEqual((folder / "latest.png").read_bytes(), b"png-data")
            self.assertEqual(image.settings, ("PNG", 1))
            self.assertGreaterEqual(seconds, 0.0)
            self.assertEqual(list(folder.glob("*.tmp.png")), [])

    def test_preview_child_and_parent_do_not_double_count_preview_stage(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio._set_progress = lambda percent, label: None
        studio._append_log = lambda line: None
        studio._render_resources = lambda: None
        studio._gpu_memory_snapshot = lambda: None
        studio.timings = {}
        studio.vae_first = studio.vae_started = studio.qwen_first = studio.qwen_started = None
        studio.train_first = None
        studio.sec_per_step = studio.eta_seconds = None
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "preview.py"
            script.write_text("print('TIMING preview_nf4_load=1.25s', flush=True)\n", encoding="utf-8")
            studio._run_process([str(script)], os.environ.copy(), "preview", 1)
        self.assertGreaterEqual(studio.timings["preview_total"], 0.0)
        self.assertLess(studio.timings["preview_total"], 1.0)
        self.assertEqual(studio.timings["preview_nf4_load"], 1.25)

    def test_run_config_includes_hub_and_drive_settings(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        values = {
            "steps":500, "rank":16, "test_mode":True, "lr":0.0003, "trigger":" Sili ",
            "resolution":512, "preview_resolution":512, "preview_cleanup_cache":True,
            "max_seq":128, "save_every":50, "seed":42,
            "attention_only":True, "grad_ckpt":True, "push":True, "private":True,
            "keep":2, "hub_id":" user/model ", "drive":True,
            "caption":"illustration of Sili", "use_txt":True, "preview_enabled":False,
        }
        for name, value in values.items():
            setattr(studio, name, SimpleNamespace(value=value))
        config = studio._config()
        self.assertEqual(config["steps"], 20)
        self.assertEqual(config["rank"], 8)
        self.assertEqual(config["hub_id"], "user/model")
        self.assertIs(config["drive"], True)

    def test_notebook_is_portable_and_backend_compiles(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len(notebook["cells"]), 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "cell", "exec")
        backend = "".join(notebook["cells"][2]["source"])
        assignments = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(backend).body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {"PRECACHE_SOURCE", "TRAIN_SOURCE", "PREVIEW_SOURCE"}
        }
        for source in assignments.values():
            compile(source, "backend", "exec")
        self.assertIn("load_in_4bit=True", assignments["PRECACHE_SOURCE"])
        self.assertIn("a.steps%a.save_every", assignments["TRAIN_SOURCE"])
        self.assertIn("raise SystemExit(75)", assignments["TRAIN_SOURCE"])
        self.assertIn("caption_embeddings", assignments["PRECACHE_SOURCE"])
        self.assertIn("preview_payload", assignments["PRECACHE_SOURCE"])
        self.assertIn("snapshot_download", assignments["TRAIN_SOURCE"])
        self.assertIn("transformer_nf4_load", assignments["TRAIN_SOURCE"])
        self.assertIn("train_steady_sec_per_step", assignments["TRAIN_SOURCE"])
        self.assertIn("--preview-at-checkpoints", assignments["TRAIN_SOURCE"])
        self.assertIn("PREVIEW_REQUIRED", assignments["TRAIN_SOURCE"])
        self.assertIn("--preview-config", assignments["PRECACHE_SOURCE"])
        self.assertIn("OzzyGT/Krea_2_Turbo_bnb_nf4", assignments["PREVIEW_SOURCE"])
        self.assertIn("AutoencoderKLQwenImage.from_pretrained", assignments["PREVIEW_SOURCE"])
        self.assertIn("tokenizer=None, text_encoder=None", assignments["PREVIEW_SOURCE"])
        self.assertNotIn("TIMING preview_total=", assignments["PREVIEW_SOURCE"])
        self.assertIn("prompt_embeds_mask", assignments["PREVIEW_SOURCE"])
        self.assertIn("num_inference_steps=8", assignments["PREVIEW_SOURCE"])
        compile(assignments["PREVIEW_SOURCE"], "preview", "exec")
        self.assertEqual(assignments["PREVIEW_SOURCE"], Path(__file__).with_name("krea2_turbo_preview.py").read_text(encoding="utf-8"))
        gui_cell = "".join(notebook["cells"][3]["source"])
        self.assertEqual(gui_cell, GUI.read_text(encoding="utf-8"))
        self.assertIn('id="grad_ckpt"', gui_cell)
        self.assertIn("character appearance and style", gui_cell)
        self.assertIn('function config()', gui_cell)
        trainer_tree = ast.parse(assignments["TRAIN_SOURCE"])
        checkpoint_fn = next(n for n in trainer_tree.body
                             if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint")
        upload_started_writes = [n.lineno for n in ast.walk(checkpoint_fn)
                                 if isinstance(n, ast.Name) and n.id == "upload_started"
                                 and isinstance(n.ctx, ast.Store)]
        upload_started_reads = [n.lineno for n in ast.walk(checkpoint_fn)
                                if isinstance(n, ast.Name) and n.id == "upload_started"
                                and isinstance(n.ctx, ast.Load)]
        self.assertTrue(upload_started_writes, "Checkpoint upload timer must be initialized locally.")
        self.assertTrue(upload_started_reads, "Checkpoint upload timing must be emitted.")
        self.assertLess(min(upload_started_writes), min(upload_started_reads))
        notebook_text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("hf_jCZg", notebook_text)
        self.assertIn("cloudflared", notebook_text)
        self.assertIn("chunkSize=8*1024*1024", notebook_text)
        self.assertIn("x.timeout=300000", notebook_text)
        self.assertNotIn("Smoke test", notebook_text)
        self.assertNotIn("sili_krea2_test", notebook_text)
        self.assertNotIn("ipywidgets", notebook_text)
        self.assertIn("skipping pip install", notebook_text)

        source = assignments["PRECACHE_SOURCE"]
        key_fn = next(n for n in ast.parse(source).body
                      if isinstance(n, ast.FunctionDef) and n.name == "caption_key")
        import hashlib
        key_ns = {"hashlib":hashlib}
        exec(compile(ast.Module(body=[key_fn], type_ignores=[]), "caption_key", "exec"), key_ns)
        key = key_ns["caption_key"]
        self.assertEqual(key("krea/Krea-2-Raw", 128, "Sili"), key("krea/Krea-2-Raw", 128, "Sili"))
        self.assertNotEqual(key("krea/Krea-2-Raw", 128, "Sili"), key("krea/Krea-2-Raw", 128, "Sili portrait"))
        self.assertNotEqual(key("krea/Krea-2-Raw", 128, "Sili"), key("krea/Krea-2-Raw", 192, "Sili"))

    def test_attached_zip_ingest_and_cache_signature(self):
        if not DATASET.is_file():
            self.skipTest("Optional private training dataset is not included in the public source package.")
        KreaStudio = load_class()
        with tempfile.TemporaryDirectory() as tmp:
            studio = KreaStudio.__new__(KreaStudio)
            studio._new_dataset_dir = lambda: Path(tmp) / "dataset"
            (Path(tmp) / "dataset").mkdir()
            studio._append_log = lambda line: None
            studio._set_status = lambda title, detail: None
            studio.dataset_summary = SimpleNamespace(value="")
            studio._ingest_files([(DATASET.name, DATASET.read_bytes())])
            self.assertEqual(len(studio._images(studio.dataset_dir)), 27)
            self.assertEqual(len(list(studio.dataset_dir.glob("*.txt"))), 27)
            self.assertIn("27 valid images", studio.dataset_summary.value)
            config = {"resolution": 512, "max_seq": 128, "trigger": "Sili",
                      "caption": "character appearance and style", "use_txt": True}
            first = studio._cache_signature(config)
            config["resolution"] = 768
            self.assertNotEqual(first, studio._cache_signature(config))
            config["resolution"] = 512
            config["preview_slots"] = [{"slot":1,"prompt":"Sili portrait"}]
            with_preview = studio._cache_signature(config)
            config["preview_slots"][0]["prompt"] = "Sili landscape"
            self.assertNotEqual(with_preview, studio._cache_signature(config))

    def test_rejects_bad_dataset_and_duplicate_names(self):
        KreaStudio = load_class()
        with tempfile.TemporaryDirectory() as tmp:
            studio = KreaStudio.__new__(KreaStudio)
            studio._new_dataset_dir = lambda: Path(tmp) / "dataset"
            (Path(tmp) / "dataset").mkdir()
            studio.dataset_summary = SimpleNamespace(value="")
            studio._append_log = lambda line: None
            studio._set_status = lambda title, detail: None
            with self.assertRaisesRegex(ValueError, "No JPG"):
                studio._ingest_files([("empty.txt", b"caption")])
        with tempfile.TemporaryDirectory() as tmp:
            studio._new_dataset_dir = lambda: Path(tmp) / "dataset"
            (Path(tmp) / "dataset").mkdir()
            image = io.BytesIO()
            Image.new("RGB", (2, 2)).save(image, format="PNG")
            with self.assertRaisesRegex(ValueError, "Duplicate filename"):
                studio._ingest_files([("a.png", image.getvalue()), ("a.png", image.getvalue())])

    def test_progress_parses_stage_output(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        studio.vae_first = studio.vae_started = studio.qwen_first = studio.qwen_started = None
        studio.train_first = None
        studio.stage_started = time.monotonic() - 3
        studio.sec_per_step = studio.eta_seconds = None
        events = []
        studio._set_progress = lambda percent, label: events.append((percent, label))
        studio._append_log = lambda line: None
        studio._process_line("  13/27 Sili_013.png -> 512x512", "cache", 27)
        studio._process_line("  caption 9/26 -> 1 image(s)", "cache", 26)
        studio._process_line("step 12/20 | loss 0.1234 | 1.25s/step | ETA 10.0s", "train", 20)
        self.assertEqual(events[-1], (60.0, "Training · 12/20"))
        self.assertGreater(events[1][0], events[0][0])
        self.assertEqual(studio.sec_per_step, 1.25)
        self.assertEqual(studio.eta_seconds, 10.0)
        self.assertGreater(studio.timings["train_cold_start"], 0)

    def test_stage_timing_events_are_stored_and_used_for_eta_speed(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        studio._append_log = lambda line: None
        studio._set_progress = lambda percent, label: None
        studio.sec_per_step = studio.eta_seconds = None
        studio._process_line("TIMING transformer_download=12.50s", "train", 20)
        studio._process_line("TIMING transformer_nf4_load=87.25s", "train", 20)
        studio._process_line("TIMING train_steady_sec_per_step=6.75s", "train", 20)
        self.assertEqual(studio.timings["transformer_download"], 12.5)
        self.assertEqual(studio.timings["transformer_nf4_load"], 87.25)
        self.assertEqual(studio.timings["sec_per_step"], 6.75)
        self.assertEqual(studio.sec_per_step, 6.75)

    def test_subprocess_stage_streams_progress_and_reports_failure(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        progress = []
        studio._set_progress = lambda percent, label: progress.append((percent, label))
        studio._append_log = lambda line: None
        studio._render_resources = lambda: None
        studio.timings = {}
        studio.vae_first = studio.vae_started = studio.qwen_first = studio.qwen_started = None
        studio.train_first = None
        studio.sec_per_step = studio.eta_seconds = None
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "stage.py"
            script.write_text("print('step 1/2 | loss 0.2', flush=True)\n"
                              "print('step 2/2 | loss 0.1', flush=True)\n", encoding="utf-8")
            studio._run_process([str(script)], os.environ.copy(), "train", 2)
            self.assertEqual(progress[-1], (100.0, "Training · 2/2"))
            script.write_text("import sys\nprint('failed', flush=True)\nsys.exit(7)\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                studio._run_process([str(script)], os.environ.copy(), "train", 2)

    def test_local_web_ui_serves_panel_and_protects_preview_png(self):
        tree = ast.parse(GUI.read_text(encoding="utf-8"))
        page = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "PAGE" for t in n.targets))
        server_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_web_ui")
        module = compile(ast.Module(body=[page, server_fn], type_ignores=[]), str(GUI), "exec")
        import html as html_module
        import os as os_module
        import re as re_module
        import shutil as shutil_module
        import subprocess as subprocess_module
        import threading as threading_module
        import time as time_module
        import zipfile as zipfile_module
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse
        ns = {"BaseHTTPRequestHandler":BaseHTTPRequestHandler, "ThreadingHTTPServer":ThreadingHTTPServer,
              "json":json, "html":html_module, "os":os_module, "parse_qs":parse_qs,
              "urlparse":urlparse, "re":re_module, "shutil":shutil_module,
              "subprocess":subprocess_module, "threading":threading_module, "time":time_module,
              "zipfile":zipfile_module, "Path":Path, "urllib":__import__("urllib").parse}
        exec(module, ns)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preview = root / "output" / "previews" / "slot_2" / "latest.png"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"test-png-bytes")

            class DummyStudio:
                resources = SimpleNamespace(value="<div>Live system</div>")
                def state(self):
                    return {"status":"Ready", "busy":False}
                def _render_resources(self):
                    pass
                def _root(self):
                    return root

            url, server, tunnel = ns["run_web_ui"](DummyStudio())
            try:
                base, key = url.split("/#", 1)
                page_text = urllib.request.urlopen(base, timeout=3).read().decode("utf-8")
                self.assertIn(">Configure</strong>", page_text)
                self.assertIn('id="preview-image-4"', page_text)
                request = urllib.request.Request(base + "/api/state")
                with self.assertRaises(urllib.error.HTTPError) as unauthorized:
                    urllib.request.urlopen(request, timeout=3)
                self.assertEqual(unauthorized.exception.code, 401)
                unauthorized.exception.close()
                headers = {"X-Krea-Key":key}
                state_request = urllib.request.Request(base + "/api/state", headers=headers)
                state = json.loads(urllib.request.urlopen(state_request, timeout=3).read())
                self.assertEqual(state["status"], "Ready")
                image_request = urllib.request.Request(base + "/api/preview?slot=2", headers=headers)
                self.assertEqual(urllib.request.urlopen(image_request, timeout=3).read(), b"test-png-bytes")
                bad_request = urllib.request.Request(base + "/api/preview?slot=9", headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as bad_slot:
                    urllib.request.urlopen(bad_request, timeout=3)
                self.assertEqual(bad_slot.exception.code, 400)
                bad_slot.exception.close()
            finally:
                server.shutdown()
                server.server_close()


    def test_lora_inventory_lists_final_and_completed_checkpoints_with_steps_and_sizes(self):
        KreaStudio = load_class()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/"output"; versions=output/"versions"
            versions.mkdir(parents=True)
            (versions/"lora_step_00050.safetensors").write_bytes(b"a"*12)
            (versions/"lora_step_00900.safetensors").write_bytes(b"b"*24)
            (versions/"lora_step_01000.safetensors").write_bytes(b"")
            (versions/"not-a-checkpoint.safetensors").write_bytes(b"ignore")
            (output/"pytorch_lora_weights.safetensors").write_bytes(b"final"*10)
            (output/"training_config.json").write_text(json.dumps({"steps":900}),encoding="utf-8")
            studio=KreaStudio.__new__(KreaStudio)
            studio._root=lambda:root
            studio.results=SimpleNamespace(value="")
            files=studio.list_loras()
            self.assertEqual([item["filename"] for item in files],[
                "pytorch_lora_weights.safetensors","lora_step_00900.safetensors","lora_step_00050.safetensors"])
            self.assertEqual(files[0]["label"],"Final · 900 steps")
            self.assertEqual(files[1]["label"],"Checkpoint · 900 steps")
            self.assertEqual(files[2]["size_bytes"],12)
            self.assertEqual(studio.resolve_lora_file(files[1]["filename"]).read_bytes(),b"b"*24)
            with self.assertRaisesRegex(ValueError,"saved-files list"):
                studio.resolve_lora_file("../private.safetensors")
            studio.refresh_results()
            self.assertIn("3 LoRA file(s)",studio.results.value)

    def test_lora_file_download_streams_while_training_and_refreshes_inventory(self):
        KreaStudio=load_class()
        tree=ast.parse(GUI.read_text(encoding="utf-8"))
        page=next(n for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=="PAGE" for t in n.targets))
        server_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_web_ui")
        module=compile(ast.Module(body=[page,server_fn],type_ignores=[]),str(GUI),"exec")
        import html as html_module
        import shutil as shutil_module
        import subprocess as subprocess_module
        import threading as threading_module
        import time as time_module
        import zipfile as zipfile_module
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from urllib.parse import parse_qs,urlparse
        ns={"BaseHTTPRequestHandler":BaseHTTPRequestHandler,"ThreadingHTTPServer":ThreadingHTTPServer,
            "json":json,"html":html_module,"os":os,"parse_qs":parse_qs,"urlparse":urlparse,
            "re":re,"shutil":shutil_module,"subprocess":subprocess_module,"threading":threading_module,
            "time":time_module,"zipfile":zipfile_module,"Path":Path,"urllib":__import__("urllib").parse}
        exec(module,ns)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/"output"; versions=output/"versions"; versions.mkdir(parents=True)
            final_bytes=b"final-adapter-bytes"*4
            final=output/"pytorch_lora_weights.safetensors"; final.write_bytes(final_bytes)
            checkpoint=versions/"lora_step_00900.safetensors"; checkpoint.write_bytes(b"checkpoint-900")
            (output/"training_config.json").write_text(json.dumps({"steps":900}),encoding="utf-8")
            studio=KreaStudio.__new__(KreaStudio); studio._root=lambda:root
            studio.results=SimpleNamespace(value=""); studio._lora_inventory=[]
            studio.busy=True
            studio.state=lambda:{"status":"Training","busy":True}
            studio.resources=SimpleNamespace(value="<div>Live system</div>")
            studio._render_resources=lambda:None
            url,server,tunnel=ns["run_web_ui"](studio)
            try:
                base,key=url.split("/#",1); headers={"X-Krea-Key":key}
                listing_request=urllib.request.Request(base+"/api/loras",headers=headers)
                listing=json.loads(urllib.request.urlopen(listing_request,timeout=3).read())
                self.assertEqual([x["filename"] for x in listing["files"]],
                                 ["pytorch_lora_weights.safetensors","lora_step_00900.safetensors"])
                unauthorized=urllib.request.Request(base+"/api/loras")
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(unauthorized,timeout=3)
                self.assertEqual(denied.exception.code,401); denied.exception.close()

                ticket_request=urllib.request.Request(base+"/api/download-ticket",data=json.dumps({
                    "kind":"file","filename":final.name}).encode(),method="POST",
                    headers={**headers,"Content-Type":"application/json"})
                ticket=json.loads(urllib.request.urlopen(ticket_request,timeout=3).read())
                self.assertEqual(ticket["filename"],final.name)
                # Direct native download starts and completes independently of the busy training worker.
                response=urllib.request.urlopen(base+"/api/download?ticket="+ticket["ticket"],timeout=3)
                self.assertEqual(response.status,200)
                self.assertEqual(int(response.headers["Content-Length"]),len(final_bytes))
                self.assertIn('filename="pytorch_lora_weights.safetensors"',response.headers["Content-Disposition"])
                self.assertEqual(response.read(),final_bytes)
                with self.assertRaises(urllib.error.HTTPError) as replay:
                    urllib.request.urlopen(base+"/api/download?ticket="+ticket["ticket"],timeout=3)
                self.assertEqual(replay.exception.code,404); replay.exception.close()

                bad_request=urllib.request.Request(base+"/api/download-ticket",data=json.dumps({
                    "kind":"file","filename":"../../secrets.txt"}).encode(),method="POST",
                    headers={**headers,"Content-Type":"application/json"})
                with self.assertRaises(urllib.error.HTTPError) as bad:
                    urllib.request.urlopen(bad_request,timeout=3)
                self.assertEqual(bad.exception.code,400); bad.exception.close()

                archive_request=urllib.request.Request(base+"/api/download-ticket",data=json.dumps({"kind":"all"}).encode(),
                    method="POST",headers={**headers,"Content-Type":"application/json"})
                archive_ticket=json.loads(urllib.request.urlopen(archive_request,timeout=3).read())
                archive_response=urllib.request.urlopen(base+"/api/download?ticket="+archive_ticket["ticket"],timeout=3)
                archive_bytes=archive_response.read()
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    self.assertEqual(set(archive.namelist()),{final.name,checkpoint.name})
                    self.assertEqual(archive.read(final.name),final_bytes)

                (versions/"lora_step_01000.safetensors").write_bytes(b"later checkpoint")
                refresh_request=urllib.request.Request(base+"/api/loras",headers=headers)
                refreshed=json.loads(urllib.request.urlopen(refresh_request,timeout=3).read())
                self.assertEqual(refreshed["files"][1]["filename"],"lora_step_01000.safetensors")
            finally:
                server.shutdown(); server.server_close()
                if tunnel is not None:tunnel.terminate()

    def test_cache_fingerprint_uses_file_content_and_cache_validation_checks_every_artifact(self):
        KreaStudio = load_class()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            image = dataset / "sample.png"
            image.write_bytes(b"same")
            studio = KreaStudio.__new__(KreaStudio)
            studio.dataset_dir = dataset
            config = {"resolution":512, "max_seq":128, "trigger":"Sili",
                      "caption":"illustration of Sili", "use_txt":True}
            first = studio._cache_signature(config)
            stat = image.stat()
            image.write_bytes(b"diff")
            os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(image.stat().st_size, stat.st_size)
            self.assertEqual(image.stat().st_mtime_ns, stat.st_mtime_ns)
            self.assertNotEqual(first, studio._cache_signature(config))

            cache = root / "cache"
            cache.mkdir()
            marker = {"signature":studio._cache_signature(config)}
            (cache / "studio_signature.json").write_text(json.dumps(marker), encoding="utf-8")
            (cache / "metadata.json").write_text(json.dumps({"items":[
                {"latent":"a.pt","embed":"b.pt","mask":"c.pt"}]}), encoding="utf-8")
            for filename in ("a.pt","b.pt","c.pt"):
                (cache / filename).write_bytes(b"x")
            self.assertTrue(studio._cache_complete(cache, marker["signature"]))
            (cache / "b.pt").unlink()
            self.assertFalse(studio._cache_complete(cache, marker["signature"]))

    def test_resumable_upload_accepts_exact_chunk_retries_without_duplicate_bytes(self):
        tree = ast.parse(GUI.read_text(encoding="utf-8"))
        page = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "PAGE" for t in n.targets))
        server_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_web_ui")
        module = compile(ast.Module(body=[page, server_fn], type_ignores=[]), str(GUI), "exec")
        import html as html_module
        import shutil as shutil_module
        import threading as threading_module
        import zipfile as zipfile_module
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            def safe_path(value):
                return temp_root if str(value) == "/content" else Path(value)

            class DummyStudio:
                resources = SimpleNamespace(value="<div>Live system</div>")
                dataset_path = SimpleNamespace(value="")
                received_path = None
                ingest_count = 0
                def state(self):
                    return {"status":"Ready", "busy":False}
                def _render_resources(self):
                    pass
                def _use_path(self):
                    self.received_path = Path(self.dataset_path.value)
                    self.ingest_count += 1

            ns = {"BaseHTTPRequestHandler":BaseHTTPRequestHandler, "ThreadingHTTPServer":ThreadingHTTPServer,
                  "json":json, "html":html_module, "os":os, "hashlib":__import__("hashlib"),
                  "parse_qs":parse_qs, "urlparse":urlparse, "re":re, "shutil":shutil_module,
                  "subprocess":__import__("subprocess"), "threading":threading_module, "time":time,
                  "zipfile":zipfile_module, "Path":safe_path, "urllib":__import__("urllib").parse}
            exec(module, ns)
            studio = DummyStudio()
            url, server, tunnel = ns["run_web_ui"](studio)
            try:
                base, key = url.split("/#", 1)
                def post(offset, payload):
                    request = urllib.request.Request(
                        base + "/api/dataset/upload", data=payload, method="POST",
                        headers={"X-Krea-Key":key, "X-Krea-Config":"%7B%7D",
                                 "X-Filename":"dataset.zip", "X-Upload-Id":"session_1234",
                                 "X-Upload-Offset":str(offset), "X-Upload-Total":"10"})
                    return json.loads(urllib.request.urlopen(request, timeout=5).read())
                self.assertEqual(post(0, b"abcde")["received"], 5)
                self.assertEqual(post(0, b"abcde")["received"], 5)
                self.assertEqual(post(5, b"fghij")["received"], 10)
                post(5, b"fghij")
                self.assertEqual(studio.ingest_count, 1)
                self.assertEqual(studio.received_path.read_bytes(), b"abcdefghij")
            finally:
                server.shutdown()
                server.server_close()
                if tunnel is not None:
                    tunnel.terminate()

    def test_live_eta_counts_down_between_training_log_lines(self):
        KreaStudio = load_class()
        studio = KreaStudio.__new__(KreaStudio)
        studio.timings = {}
        studio.eta_seconds = 20.0
        studio.busy = True
        studio.current_stage = "train"
        studio.stage_started = 95.0
        studio.train_first = 100.0
        studio.last_step_at = 110.0
        studio.sec_per_step = 2.0
        studio.cache_started = studio.vae_first = studio.vae_started = None
        studio.qwen_started = studio.qwen_first = None
        with patch("time.monotonic", return_value=115.0):
            html = studio.timing_html()
        self.assertIn("Train ETA · live", html)
        self.assertIn("<b>15s</b>", html)

    def test_checkpoint_discovery_recovers_previous_and_skips_corrupt_checkpoint(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        backend = "".join(notebook["cells"][2]["source"])
        assignments = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(backend).body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "TRAIN_SOURCE"
        }
        tree = ast.parse(assignments["TRAIN_SOURCE"])
        finder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "newest_checkpoint")
        ns = {"Path":Path, "json":json, "os":os}
        exec(compile(ast.Module(body=[finder], type_ignores=[]), "newest_checkpoint", "exec"), ns)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / ".checkpoint-10.previous"
            backup.mkdir()
            (backup / "training_state.pt").write_bytes(b"state-10")
            (backup / "pytorch_lora_weights.safetensors").write_bytes(b"adapter-10")
            (backup / "checkpoint.json").write_text(json.dumps({
                "step":10, "state_bytes":8, "adapter_bytes":10
            }), encoding="utf-8")
            corrupt = root / "checkpoint-20"
            corrupt.mkdir()
            state = corrupt / "training_state.pt"
            adapter = corrupt / "pytorch_lora_weights.safetensors"
            state.write_bytes(b"state-20")
            adapter.write_bytes(b"adapter-20")
            (corrupt / "checkpoint.json").write_text(json.dumps({
                "step":20, "state_bytes":999, "adapter_bytes":adapter.stat().st_size
            }), encoding="utf-8")
            self.assertEqual(ns["newest_checkpoint"](root), root / "checkpoint-10")
            self.assertFalse(backup.exists())

    def test_generated_backend_includes_reusable_latents_and_atomic_resume_checkpoints(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = "".join(notebook["cells"][2]["source"])
        assignments = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(cell).body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {"PRECACHE_SOURCE", "TRAIN_SOURCE", "PREVIEW_SOURCE"}
        }
        precache = assignments["PRECACHE_SOURCE"]
        trainer = assignments["TRAIN_SOURCE"]
        preview = assignments["PREVIEW_SOURCE"]
        self.assertIn("OzzyGT/Krea_2_Turbo_bnb_nf4", preview)
        self.assertIn('TURBO_MODEL_REVISION = "5458debf8356a6646a5aa814de28dcea881f8a6d"', preview)
        self.assertIn("revision=TURBO_MODEL_REVISION", preview)
        self.assertIn("num_inference_steps=8", preview)
        self.assertIn("ThreadPoolExecutor(max_workers=1", preview)
        self.assertEqual(preview.count("Krea2Transformer2DModel.from_pretrained("), 1)
        self.assertEqual(preview.count("AutoencoderKLQwenImage.from_pretrained("), 1)
        self.assertEqual(preview.count("Krea2Pipeline.from_pretrained("), 1)
        self.assertIn("save_executor.submit(save_preview_files", preview)
        self.assertIn('"status": "saving"', preview)
        self.assertNotIn('pipe.to("cpu")', preview)
        self.assertIn("torch.cuda.synchronize()", preview)
        self.assertIn("AutoencoderKLQwenImage.from_pretrained", preview)
        self.assertIn("local_files_only=True", preview)
        self.assertIn("tokenizer=None, text_encoder=None", preview)
        self.assertNotIn("TIMING preview_total=", preview)
        self.assertIn("PREVIEW_MODEL_PATTERNS", GUI.read_text(encoding="utf-8"))
        self.assertIn("latent-cache-v2", precache)
        self.assertIn("latent_check.ndim==4 and latent_check.shape[0]==1", precache)
        self.assertIn("LATENT_CACHE", precache)
        self.assertIn("os.replace(temp_latent,cached_latent)", precache)
        self.assertIn("checkpoint.json", trainer)
        self.assertIn("latest_checkpoint.json", trainer)
        self.assertIn(".checkpoint-{step}.pending", trainer)
        self.assertIn("os.replace(tmp_ck,ck)", trainer)
        self.assertIn("version_tmp=versions/f'.{version_file.name}.pending'", trainer)
        self.assertIn("os.replace(version_tmp,version_file)", trainer)
        self.assertIn("def newest_checkpoint(out):", trainer)
        compile(precache, "precache", "exec")
        compile(trainer, "trainer", "exec")
        compile(preview, "preview", "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)


