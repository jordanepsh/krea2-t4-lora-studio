# Validation record

## Local CPU validation

Run from this directory:

```powershell
python -m pip install Pillow
python -m unittest -v test_krea2_gui.py
```

The suite exercises notebook portability and compilation, embedded source parity, cache and checkpoint helpers, dataset validation, preview configuration, local Studio HTTP behavior, saved-LoRA inventory/refresh, traversal rejection, single-use download tickets, ZIP creation, and exact binary streaming while a training stage is marked busy. The optional dataset-ingest test is skipped when the private sample ZIP is absent.

## Colab/T4 validation still required for each release

1. Run setup in a fresh T4 runtime and confirm the Studio link opens.
2. Use a small dataset and **Quick run** to validate cache preparation and a short training run.
3. During a longer run, refresh **Saved LoRA files** and download a completed checkpoint before training ends. Compare the downloaded byte size with the Studio row.
4. After completion, download the final adapter and **Download all** ZIP. Confirm the final adapter opens in the intended generator.
5. If Turbo previews are enabled, confirm preview cards update, VRAM returns to the training baseline, and training resumes after each checkpoint.

Local test success is not a substitute for these live runtime checks. Record Colab interruptions separately from measured trainer stage timings.
