# Validation record

## Local CPU validation

Run from this directory:

```powershell
python -m pip install Pillow
python -m unittest -v test_krea2_gui.py
```

The suite exercises notebook portability and compilation, embedded source parity, cache and checkpoint helpers, dataset validation, preview configuration, local Studio HTTP behavior, saved-LoRA inventory/refresh, traversal rejection, single-use download tickets, ZIP creation, and exact binary streaming while a training stage is marked busy. The optional dataset-ingest test is skipped when the private sample ZIP is absent.

Current local result (28 Sep 2026): **28 tests run, 27 passed, 1 skipped** because the private sample dataset is not included in the public source package.

## Maintainer-reported Colab/T4 validation

The maintainer reports a successful end-to-end Colab/T4 test: cache preparation, training, preview generation, and LoRA output/download completed successfully. The report is based on a user-run session; Colab runtime and network conditions can affect timings. Approximate observations are summarized in [BENCHMARKS.md](BENCHMARKS.md).

For future releases, repeat these checks in a fresh runtime:

1. Run setup in a fresh T4 runtime and confirm the Studio link opens.
2. Use a small dataset and **Quick run** to validate cache preparation and a short training run.
3. During a longer run, refresh **Saved LoRA files** and download a completed checkpoint before training ends. Compare the downloaded byte size with the Studio row.
4. After completion, download the final adapter and **Download all** ZIP. Confirm the final adapter opens in the intended generator.
5. If Turbo previews are enabled, confirm preview cards update, VRAM returns to the training baseline, and training resumes after each checkpoint.

Local test success is not a substitute for live runtime checks. Record Colab interruptions separately from measured trainer stage timings.

## Continuous integration

GitHub Actions runs the same CPU-only regression suite on Python 3.10 through 3.13 for pushes, pull requests, and manual runs. A green CI run confirms those checks on the listed Python versions; it does not replace the Colab/T4 checks above.
