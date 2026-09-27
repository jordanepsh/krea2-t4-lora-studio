# Approximate T4 timings

These figures summarize maintainer-reported Google Colab T4 runs of Krea 2 LoRA Studio. They are useful as rough planning estimates, not controlled benchmarks or promises. Colab reconnects, network interruptions, model cache state, dataset settings, and preview settings all change elapsed time.

## Observed examples

| Operation | Approximate observation | Context |
| --- | ---: | --- |
| Cold start | about 8 minutes | User-corrected estimate for an uninterrupted run; longer logs included reconnect delays. |
| Training segment | 100 steps in about 14 min 52 sec | One reported run. This includes that run's observed elapsed time and should not be treated as a universal step rate. |
| Training speed | about 7–8 sec/step | Approximate figures shown by separate run logs; settings and interruptions differed. |
| Two Turbo previews | about 82 seconds total | Reported preview process time, including model setup/loading and generating two images. Image generation itself was about 44 seconds in that run. |

## Reading the numbers

- Earlier cold-start and total-training figures were inflated by Colab disconnects and reconnects. They are not reliable measurements of uninterrupted compute time.
- A full-run wall-clock estimate cannot be derived reliably from those interrupted sessions. Use the figures above only to plan roughly, then measure a fresh run in your own runtime.
- Preview time depends on whether Turbo weights are already cached and on image resolution and count. The first preview can include the one-time model load.
- The Studio reports stage timings and step speed to help users compare runs under their own conditions.
