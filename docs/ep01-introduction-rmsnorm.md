# Episode 1: Introduction - Bielik Architecture, Pallas/TPU Mental Model, RMSNorm

[Back to README](../README.md) | [Triton/GPU counterpart](https://github.com/qooba/bielik-anatomy-triton/blob/main/docs/ep03-rmsnorm-softmax-fused.md)

---

## Overview

This episode moves the [`bielik-anatomy-triton`](https://github.com/qooba/bielik-anatomy-triton)
series from custom Triton GPU kernels to Google Cloud TPUs, using JAX and Pallas.

**Target model:** Bielik-1.5B - 1.6B parameters, 32 decoder layers, hidden size 1536,
Grouped Query Attention (12 query heads, 2 KV heads), SwiGLU MLP with intermediate
size 8960, and a bias term on every linear layer.

**The stack:**
- [JAX](https://docs.jax.dev/) - Google's high-performance computing framework,
  built on the XLA compiler, which traces, optimizes, and fuses operations down to
  TPU machine instructions.
- [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) - JAX's kernel
  language, targeting [Mosaic](https://docs.jax.dev/en/latest/pallas/tpu/index.html),
  the low-level compiler backend that speaks directly to TPU hardware. It bypasses
  XLA's own heuristics for hand-crafted memory transfers and compute patterns.

This episode's kernel, RMSNorm, is memory-bound rather than compute-bound, which
makes it a good first test of a simple question: does a hand-written Pallas kernel
beat XLA's own fusion on a TPU the way hand-written Triton kernels beat plain
PyTorch on a GPU?

## Relevant Code

- [`kernels/normalization/rms_norm.py`](/kernels/normalization/rms_norm.py) -
  `rms_norm_ref()` (plain-JAX reference) and `rms_norm_pallas()` (fused Pallas
  kernel), side by side in one file.
- [`tests/test_rms_norm.py`](/tests/test_rms_norm.py) - correctness only, no
  timing. Runs everywhere via Pallas `interpret=True` (free, no TPU); the real-TPU
  pass runs automatically too when a TPU device is visible.
- [`benchmarks/normalization/benchmark_rms_norm.py`](/benchmarks/normalization/benchmark_rms_norm.py) -
  `pallas` (hand-written kernel, `block_rows=128`) vs `jax_native` (plain ops,
  `jax.jit`-compiled - XLA fuses it). Two sweeps:
  - bandwidth vs `hidden_size` (`n_rows=512` fixed)
  - bandwidth vs `n_rows` (`hidden_size=1536` fixed, the real Bielik dimension)

  plus a Bielik-config (`hidden_size=1536`) summary table.

```bash
# Local, no TPU needed
PYTHONPATH=. python3 tests/test_rms_norm.py

# Real TPU
make benchmark-rms-norm
```

Or run [`notebooks/normalization/rms_norm.ipynb`](/notebooks/normalization/rms_norm.ipynb)
directly in [Google Colab](https://colab.research.google.com/github/qooba/bielik-on-tpu/blob/main/notebooks/normalization/rms_norm.ipynb)
on a free TPU runtime - no local setup at all.

### How the kernel works

`rms_norm_pallas()` defines a 1D grid over row blocks via `pl.cdiv`. `BlockSpec`
maps each grid step `i` to a slice of memory through an index function: for `x`,
`lambda i: (i, 0)` picks the `i`-th row block while keeping the full hidden
dimension; for `w`, `lambda i: (0,)` reuses the same weight vector on every tile.
`n_rows` doesn't need to be a multiple of `block_rows` - the grid uses
`pl.cdiv`, so Pallas masks the partial last block automatically, and since RMSNorm
normalizes each row independently, padding rows can't corrupt real ones.

The `interpret=True` flag runs the kernel body as plain JAX ops on CPU, no Mosaic
compilation - useful for debugging and testing kernel logic with zero TPU cost
before ever touching real hardware.

### Where this runs

Real-hardware numbers below were measured on a Google Cloud TPU v5e-1 via Google
Colab, with the installation pinned to `jax[tpu]==0.11.1` for reproducible Mosaic
lowering.

## Results on v5e-1 (via `make benchmark-rms-norm`)

<p align="center">
    <img src="/docs/plots/normalization/rmsnorm-bandwidth-vs-rows.png" alt="rmsnorm-bandwidth-vs-rows" style="max-width: 100%;">
</p>

Sweeping `n_rows` from 1 up to 1024 at Bielik's hidden size of 1536: across most of
the range Pallas and native JAX track each other closely, both scaling up toward
60-70 GB/s, with native JAX actually edging ahead at the largest batch sizes.

<p align="center">
    <img src="/docs/plots/normalization/rmsnorm-bandwidth-vs-hidden-size.png" alt="rmsnorm-bandwidth-vs-hidden-size" style="max-width: 100%;">
</p>

Sweeping `hidden_size` up to 4096 tells the same story - the curves remain tightly
grouped around Bielik's actual hidden size of 1536 and beyond, both saturating near
80 GB/s.

| n_rows | Provider | GB/s |
|---|---|---|
| 512 | Pallas (fused) | 31.0 |
| 512 | JAX Native (jit-fused) | 27.6 |

**Takeaway:** for a 1D reduction in a memory-bound layer like RMSNorm, XLA's
automatic fusion is already extremely competitive with a hand-written Pallas
kernel. Custom Pallas isn't about beating XLA here - it's about taking full
control over tiling and memory layout for the harder, compute-bound kernels ahead
(starting with matrix multiplication next episode).

Methodology note: the benchmark loop uses `time.perf_counter` synchronized with
`jax.block_until_ready`, so these are end-to-end (host dispatch + device
execution) numbers, not a pure on-chip trace. A deeper xprof-based profiling pass
comes in a later episode.
