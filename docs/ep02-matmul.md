# Episode 2: Matmul on TPU - Pallas vs JAX Native, with Autotuning

[Back to README](../README.md) | [Triton/GPU counterpart](https://github.com/qooba/bielik-anatomy-triton/blob/main/docs/ep02-matmul.md)

---

## Overview

Episode 1 built RMSNorm - memory-bound, one row independent of the next, and a
near wash between hand-written Pallas and plain JAX. This episode moves to
matmul: compute-bound, a genuine reduction axis, and the TPU's systolic Matrix
Multiply Unit (MXU) - the actual hardware every projection and FFN layer in
Bielik runs on.

`M x K` times `K x N` is `2*M*K*N` FLOPs against `(M*K + K*N + M*N)` elements
moved - the ratio grows with size, so a large matmul is genuinely compute-bound.
That's why this episode's charts are TFLOPS, not GB/s.

`matmul_pallas()` is structured as the idiomatic Pallas TPU pattern: a 3D grid
over `(M tiles, N tiles, K tiles)`, K innermost, with a scratch VMEM accumulator
zeroed at the first K step and written to the output at the last. Two things a
Triton kernel for the same op would do that don't carry over:

- **No `GROUP_SIZE_M` program-ID swizzling.** Triton reorders program IDs for L2
  cache locality on GPU - a concern that doesn't apply to TPU's VMEM
  double-buffering.
- **No `@triton.autotune` decorator.** Pallas has nothing built in like it -
  `block_m`/`block_n`/`block_k` default to a plain `128`. Whether that default
  costs something real (unlike RMSNorm's) is exactly what this episode measures.

## Relevant Code

- [`kernels/matmul/matmul.py`](/kernels/matmul/matmul.py) - `matmul_ref()`
  (plain-JAX) and `matmul_pallas()` (tiled Pallas kernel, VMEM accumulator)
  side by side in one file.
- [`kernels/common/precision.py`](/kernels/common/precision.py) -
  `dot_precision()`, shared by the reference and the kernel.
- [`kernels/matmul/autotune.py`](/kernels/matmul/autotune.py) - `autotune_matmul()`
  / `matmul_pallas_autotuned()`, a hand-rolled autotuning sweep (see Follow-up
  below).
- [`tests/test_matmul.py`](/tests/test_matmul.py) - correctness only, no
  timing. Shapes include non-divisible M/K/N on every axis simultaneously, a
  smaller `block_m` with valid `block_n`/`block_k`, and a NaN check (guards
  against the unmasked-K-read bug regressing).
- [`tests/test_matmul_autotune.py`](/tests/test_matmul_autotune.py) - wiring
  only for the autotuner (candidate generation stays tile-aligned, the sweep
  runs end to end and picks *some* valid config, that config's output matches
  `matmul_ref`, the in-process cache is hit on a repeat call).
- [`benchmarks/matmul/benchmark_matmul.py`](/benchmarks/matmul/benchmark_matmul.py) -
  three providers: `pallas` (fixed `block=128`), `pallas_autotuned`,
  `jax_native`. TFLOPS since matmul is compute-bound. A square-size sweep
  (`512..4096`) plus eight named Bielik-1.5B projection shapes (Q/K/FFN-up/
  FFN-down, single-token and `seq_len=128`), all in bf16 - Bielik's real
  checkpoint dtype.

```bash
# Local, no TPU needed
PYTHONPATH=. python3 tests/test_matmul.py tests/test_matmul_autotune.py

# Real TPU
make benchmark-matmul
```

Or run [`notebooks/matmul/matmul.ipynb`](/notebooks/matmul/matmul.ipynb)
directly in [Google Colab](https://colab.research.google.com/github/qooba/bielik-on-tpu/blob/main/notebooks/matmul/matmul.ipynb)
on a free TPU runtime - no local setup at all.

## Lightweight, hand-rolled autotuner

Pallas has no `@triton.autotune` decorator, but nothing stops hand-rolling the
same idea: `autotune_matmul(M, K, N, ...)` sweeps every tile-aligned `(block_m,
block_n, block_k)` combination for a given shape on real hardware, times each
with a cheap warmup/rep, and caches the fastest. `matmul_pallas_autotuned()` is
the drop-in call that uses it.

## Results on v5e-1 (via `make benchmark-matmul`)

<p align="center">
    <img src="/docs/plots/matmul/matmul-tflops-vs-size.png" alt="matmul-tflops-vs-size" style="max-width: 100%;">
</p>

Compare against the RTX 4060 Ti (Triton) numbers in
[`bielik-anatomy-triton/docs/ep02-matmul.md`](https://github.com/qooba/bielik-anatomy-triton/blob/main/docs/ep02-matmul.md).
