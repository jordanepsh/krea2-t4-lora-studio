# Contributing

Thanks for helping improve Krea 2 LoRA Studio.

## Before opening an issue

- Include the notebook/runtime stage and a short, sanitized error excerpt.
- Say whether the issue occurred in Colab or in the local CPU-only tests.
- Never attach datasets, model weights, generated images, tokens, private repository names, or active Colab/Studio links.

## Pull requests

- Keep changes focused and preserve the staged cache/train/preview process.
- Run `python -m unittest -v test_krea2_gui.py` from this directory.
- If the notebook runtime source changes, keep the embedded code and sidecar source in sync; the tests check source parity.
- Describe which hardware-dependent checks you could not run. Local tests cannot confirm gated model access or live T4 behavior.
