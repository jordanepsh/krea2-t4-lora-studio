# Krea 2 LoRA Studio

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jordanepsh/krea2-t4-lora-studio/blob/main/Krea2_T4_LoRA_Studio.ipynb)
[![Tests](https://github.com/jordanepsh/krea2-t4-lora-studio/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/jordanepsh/krea2-t4-lora-studio/actions/workflows/tests.yml)

A self-contained Google Colab notebook for training Krea 2 LoRAs on a T4 GPU. A lightweight browser Studio guides dataset setup, staged caching and training, optional Turbo checkpoint previews, resume, Hugging Face upload, and individual adapter downloads.

## Start in Google Colab

1. Open `Krea2_T4_LoRA_Studio.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Make sure your Hugging Face account has access to [`krea/Krea-2-Raw`](https://huggingface.co/krea/Krea-2-Raw). If Colab must authenticate, add a Colab Secret named `HF_TOKEN` and enable notebook access for that secret.
3. Run all notebook cells from top to bottom. Open the temporary Studio link printed at the end.
4. Set a project name and trigger word, add a ZIP dataset or a folder already in the runtime, and choose **Prepare cache**.
5. Choose **Train**. For a short setup check, enable **Quick run** first. For an existing saved run, choose **Resume**.
6. Download a final model or checkpoint from **Saved LoRA files**. Use **Refresh** to reload the list or **Download all** to make a ZIP of available adapters.

## Dataset format

Upload one ZIP containing JPG, PNG, or WEBP images. Matching `.txt` captions are optional; each caption must have the same filename stem as its image, for example `portrait.png` and `portrait.txt`. The notebook validates the archive before caching. No sample dataset or trained weights are included here.

## What the Studio does

- Uses separate cache, training, and optional preview processes to control peak GPU memory.
- Reuses cached image latents and caption embeddings when the dataset and settings match.
- Saves permanent LoRA snapshots alongside resumable training checkpoints.
- Lists each completed LoRA by checkpoint step and file size. Adapter downloads stream directly from the Studio; saved checkpoints remain downloadable during training.
- Optionally generates 8-step images using a compact, pre-quantized NF4 Krea 2 Turbo model. Preview is off by default and defaults to 512 px.
- Shows progress, timing, ETA, system readings, and preview status.
- Uploads to Hugging Face only when enabled in Studio; the repository defaults to private.

## Storage and runtime notes

The temporary Studio URL and Colab runtime stop working when the runtime ends. Files under `/content` are temporary. Enable Google Drive storage before uploading the dataset if you need cached files and resume checkpoints to persist through a runtime reset. A full runtime reset may still require rerunning setup and reconnecting the Studio.

The base model is gated. Every user needs their own Hugging Face access and token as applicable. Do not commit tokens, private datasets, model weights, generated outputs, or runtime caches. The public notebook does not contain these files or credentials.

## Maintainer checks

The sidecar Python files are included for code review and local CPU-only regression tests; the notebook embeds the same runtime code and can run standalone.

    python -m pip install Pillow
    python -m unittest -v test_krea2_gui.py

A local pass checks Python/notebook structure, the Studio API, dataset validation, and exact adapter download bytes. It does not validate live Colab hardware, gated model access, network stability, or training quality. Follow [TESTING.md](TESTING.md) for the separate Colab check.

## Contributing

Bug reports and focused improvements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request. Please do not attach datasets, model weights, generated images, access tokens, or Colab links that expose a live runtime.

## License

The original Studio source code and project documentation are licensed under the [MIT License](LICENSE). This license does not cover Krea 2 model weights, third-party packages, datasets, generated images, or LoRAs. Those remain subject to their own terms. In particular, Krea 2 Raw and Turbo are governed by the [Krea 2 Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE); see [MODEL_TERMS.md](MODEL_TERMS.md) for the project-specific scope note.
