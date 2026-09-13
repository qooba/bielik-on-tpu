# Bielik on TPU - JAX/Pallas Port of the Triton Kernel Series

A hands-on video series porting [`bielik-anatomy-triton`](https://github.com/qooba/bielik-anatomy-triton)
(custom Triton GPU kernels for the Polish LLM **Bielik 1.5B**) to Google Cloud TPUs
using [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html).

**Model:** [Bielik-1.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) (1.6B parameters, Polish)

This repository is released episode by episode, in step with the video series.

---

## Series Overview

| # | Episode | Key Result | Triton/GPU counterpart |
|---|---------|------------|-------------------------|
| 01 | [Introduction & RMSNorm](/docs/ep01-introduction-rmsnorm.md) | Bielik's architecture, the JAX/Pallas/Mosaic stack, and a fused single-pass RMSNorm Pallas TPU kernel | [ep03](https://github.com/qooba/bielik-anatomy-triton/blob/main/docs/ep03-rmsnorm-softmax-fused.md) |

More episodes land here as they're released.

## Project Structure

```
bielik-on-tpu/
├── kernels/                 # JAX Pallas TPU kernels, one file per op: jax ref + pallas kernel together
│   └── normalization/        #   rms_norm.py
├── benchmarks/               # Performance benchmarks (bandwidth/TFLOPS sweeps)
│   ├── common/                #   bench.py (do_bench), plotting.py
│   └── normalization/         #   benchmark_rms_norm.py
├── tests/                    # Correctness-only tests, runnable locally with no TPU (Pallas interpret mode)
└── docs/                     # Episode docs
```

## Constraints

- **Pure `.py` scripts, no Jupyter notebooks as source.**
- **No local TPU required for development.** Local iteration uses `jax` (CPU-only)
  with Pallas `interpret=True` to catch logic bugs for free; a real TPU is only
  needed to confirm Mosaic lowering and collect real performance numbers.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/qooba/bielik-on-tpu
cd bielik-on-tpu

# Local, TPU-free dev loop (fast, zero cost)
python3 -m venv .venv
.venv/bin/pip install jax==0.11.1 numpy matplotlib
PYTHONPATH=. .venv/bin/python tests/test_rms_norm.py

# Real TPU (e.g. a Google Colab v5e-1 runtime): run the benchmark
pip install jax[tpu]==0.11.1 numpy matplotlib
make benchmark-rms-norm
```

## Development

```bash
make install     # create .venv and install requirements.txt
make test        # run kernel tests in Pallas interpret mode (no TPU needed)
make fmt         # format with black + ruff
make fmt-check   # check formatting without modifying files
```

CI runs `ruff`/`black` checks and the kernel test suite in Pallas interpret mode on
every push and pull request (see `.github/workflows/ci.yml`) - no TPU access
needed, since interpret mode runs the exact same kernel logic on plain CPU JAX.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
