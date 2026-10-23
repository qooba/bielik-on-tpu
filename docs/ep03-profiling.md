# Episode 3: Profiling JAX Pallas on TPU - What Wall-Clock Benchmarks Can't Tell You

[Back to README](../README.md)

---

## Overview

Episodes 1 and 2 measured RMSNorm and matmul with wall-clock benchmarks -
`time.perf_counter` synced with `block_until_ready`, host dispatch plus device
execution together. This episode asks a different question: what happens when
you look *inside* the chip instead of just timing the whole round trip?

No new kernel here - this episode reuses RMSNorm (memory-bound, episode 1) and
matmul (compute-bound, episode 2) as two case studies for the profiler itself,
entirely in [`notebooks/profiling/profiling.ipynb`](/notebooks/profiling/profiling.ipynb).
Two diagnostic tools, both standard JAX/XLA, no extra dependencies:

- **`jax.profiler.trace()`** - runtime traces: isolates host dispatch from pure
  TPU execution, down to a per-op breakdown.
- **`.lower(...).compile().as_text()`** - the compiled HLO text: shows what XLA
  actually fused under the hood, as opposed to a Pallas kernel's opaque
  custom-call boundary.

## Relevant Code

- [`notebooks/profiling/profiling.ipynb`](/notebooks/profiling/profiling.ipynb) -
  the entire episode. Self-contained: it reimplements trace capture/parsing and
  the `cost_estimate` variants inline, importing only `kernels/normalization/`
  and `kernels/matmul/` from episodes 1-2 - no new `kernels`/`tests`/
  `benchmarks` code ships with this episode.

```bash
python3 -m venv .venv
.venv/bin/pip install jax[tpu]==0.11.1 numpy pandas
```

Then run [`notebooks/profiling/profiling.ipynb`](/notebooks/profiling/profiling.ipynb)
directly in [Google Colab](https://colab.research.google.com/github/qooba/bielik-on-tpu/blob/main/notebooks/profiling/profiling.ipynb)
on a free TPU runtime - profiling a real Pallas/Mosaic trace needs real
hardware, same as every benchmark in this series.


