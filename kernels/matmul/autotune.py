import itertools
import time

import jax
import jax.numpy as jnp

from kernels.common.tile_candidates import (
    block_k_candidates_when_m_lt_8,
    block_m_candidates,
    block_n_or_k_candidates,
    block_n_or_k_candidates_wide,
)
from kernels.matmul.matmul import matmul_pallas

__all__ = [
    "block_m_candidates",
    "block_n_or_k_candidates",
    "block_n_or_k_candidates_wide",
    "autotune_matmul",
    "matmul_pallas_autotuned",
    "preload",
]

_CACHE: dict[tuple, tuple[int, int, int]] = {}


def preload(entries: dict[tuple, tuple[int, int, int]]) -> None:
    _CACHE.update(entries)


_VMEM_SAFETY_LIMIT_BYTES = 15 * 1024 * 1024


def _estimated_vmem_bytes(block_m: int, block_n: int, block_k: int, itemsize: int = 4) -> int:
    a_block = block_m * block_k
    b_block = block_k * block_n
    acc_block = block_m * block_n
    out_block = block_m * block_n
    io_elems = 2 * a_block + 2 * b_block + out_block
    return io_elems * itemsize + acc_block * 4


def _time_config(
    a: jnp.ndarray,
    b: jnp.ndarray,
    bias: jnp.ndarray | None,
    block_m: int,
    block_n: int,
    block_k: int,
    interpret: bool,
    warmup: int,
    rep: int,
) -> float:
    for _ in range(warmup):
        jax.block_until_ready(matmul_pallas(a, b, bias, block_m, block_n, block_k, interpret))
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(matmul_pallas(a, b, bias, block_m, block_n, block_k, interpret))
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def autotune_matmul(
    M: int,
    K: int,
    N: int,
    dtype: jnp.dtype = jnp.float32,
    bias: bool = False,
    interpret: bool = False,
    warmup: int = 3,
    rep: int = 10,
    verbose: bool = False,
) -> tuple[int, int, int]:
    key = (M, K, N, str(jnp.dtype(dtype)), bias, interpret)
    if key in _CACHE:
        return _CACHE[key]

    key_a, key_b, key_bias = jax.random.split(jax.random.PRNGKey(0), 3)
    a = jax.random.normal(key_a, (M, K), dtype=dtype)
    b = jax.random.normal(key_b, (K, N), dtype=dtype)
    bias_arr = jax.random.normal(key_bias, (N,), dtype=dtype) if bias else None

    best_config = None
    best_time = float("inf")
    block_k_pool = (
        block_k_candidates_when_m_lt_8(K, wide=True) if M < 8 else block_n_or_k_candidates_wide(K)
    )
    for block_m, block_n, block_k in itertools.product(
        block_m_candidates(M), block_n_or_k_candidates_wide(N), block_k_pool
    ):
        est_bytes = _estimated_vmem_bytes(
            block_m, block_n, block_k, itemsize=jnp.dtype(dtype).itemsize
        )
        if est_bytes > _VMEM_SAFETY_LIMIT_BYTES:
            if verbose:
                print(
                    f"  block=({block_m},{block_n},{block_k}) SKIPPED: "
                    f"estimated {est_bytes / 1e6:.1f}MB > {_VMEM_SAFETY_LIMIT_BYTES / 1e6:.0f}MB "
                    "safety limit"
                )
            continue
        try:
            t = _time_config(
                a, b, bias_arr, block_m, block_n, block_k, interpret, warmup=warmup, rep=rep
            )
        except Exception as e:  # noqa: BLE001 -- a rejected config shouldn't abort the sweep
            if verbose:
                print(f"  block=({block_m},{block_n},{block_k}) FAILED: {e}")
            continue
        if verbose:
            print(f"  block=({block_m},{block_n},{block_k}) {t * 1e3:.3f}ms")
        if t < best_time:
            best_time = t
            best_config = (block_m, block_n, block_k)

    if best_config is None:
        raise RuntimeError(f"autotune_matmul: no valid block config found for shape ({M},{K},{N})")

    _CACHE[key] = best_config
    return best_config


def matmul_pallas_autotuned(
    a: jnp.ndarray,
    b: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    interpret: bool = False,
    warmup: int = 3,
    rep: int = 10,
    verbose: bool = False,
) -> jnp.ndarray:
    M, K = a.shape
    _, N = b.shape
    block_m, block_n, block_k = autotune_matmul(
        M,
        K,
        N,
        dtype=a.dtype,
        bias=bias is not None,
        interpret=interpret,
        warmup=warmup,
        rep=rep,
        verbose=verbose,
    )
    return matmul_pallas(a, b, bias, block_m, block_n, block_k, interpret)
