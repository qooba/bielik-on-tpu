import time

import jax
import numpy as np


def do_bench(fn, quantiles=(0.5, 0.2, 0.8), warmup=5, rep=50):
    for _ in range(warmup):
        jax.block_until_ready(fn())

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1e3)

    return tuple(float(v) for v in np.percentile(times, [q * 100 for q in quantiles]))


def device_name() -> str:
    return jax.devices()[0].device_kind


def pick_block_size(dim: int, block_size: int) -> int:
    return min(dim, block_size)


def tflops(ms: float, flop_count: int) -> float:
    return flop_count / (ms * 1e9)
