import time

import jax
import jax.numpy as jnp

from kernels.common.tile_candidates import block_m_candidates
from kernels.normalization.rms_norm import rms_norm_pallas

_CACHE: dict[tuple, int] = {}


def preload(entries: dict[tuple, int]) -> None:
    _CACHE.update(entries)


_VMEM_SAFETY_LIMIT_BYTES = 15 * 1024 * 1024


def _estimated_vmem_bytes(block_rows: int, hidden_size: int, itemsize: int = 4) -> int:
    x_or_o = block_rows * hidden_size
    total_elems = 2 * x_or_o + hidden_size
    return 2 * total_elems * itemsize


def _time_config(
    x: jnp.ndarray,
    w: jnp.ndarray,
    eps: float,
    block_rows: int,
    interpret: bool,
    warmup: int,
    rep: int,
) -> float:
    for _ in range(warmup):
        jax.block_until_ready(
            rms_norm_pallas(x, w, eps=eps, block_rows=block_rows, interpret=interpret)
        )
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(
            rms_norm_pallas(x, w, eps=eps, block_rows=block_rows, interpret=interpret)
        )
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def autotune_rms_norm(
    n_rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
    interpret: bool = False,
    warmup: int = 3,
    rep: int = 10,
    verbose: bool = False,
) -> int:
    key = (n_rows, hidden_size, eps, str(jnp.dtype(dtype)), interpret)
    if key in _CACHE:
        return _CACHE[key]

    key_x, key_w = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key_x, (n_rows, hidden_size), dtype=dtype)
    w = jax.random.normal(key_w, (hidden_size,), dtype=dtype)

    best_config = None
    best_time = float("inf")
    for block_rows in block_m_candidates(n_rows):
        est_bytes = _estimated_vmem_bytes(
            block_rows, hidden_size, itemsize=jnp.dtype(dtype).itemsize
        )
        if est_bytes > _VMEM_SAFETY_LIMIT_BYTES:
            if verbose:
                print(
                    f"  block_rows={block_rows} SKIPPED: estimated {est_bytes / 1e6:.1f}MB "
                    f"> {_VMEM_SAFETY_LIMIT_BYTES / 1e6:.0f}MB safety limit"
                )
            continue
        try:
            t = _time_config(x, w, eps, block_rows, interpret, warmup=warmup, rep=rep)
        except Exception as e:  # noqa: BLE001 -- a rejected config shouldn't abort the sweep
            if verbose:
                print(f"  block_rows={block_rows} FAILED: {e}")
            continue
        if verbose:
            print(f"  block_rows={block_rows} {t * 1e3:.3f}ms")
        if t < best_time:
            best_time = t
            best_config = block_rows

    if best_config is None:
        raise RuntimeError(f"autotune_rms_norm: no valid block_rows found for n_rows={n_rows}")

    _CACHE[key] = best_config
    return best_config


def rms_norm_pallas_autotuned(
    x: jnp.ndarray,
    w: jnp.ndarray,
    eps: float = 1e-6,
    interpret: bool = False,
    warmup: int = 3,
    rep: int = 10,
    verbose: bool = False,
) -> jnp.ndarray:
    n_rows, hidden_size = x.shape
    block_rows = autotune_rms_norm(
        n_rows,
        hidden_size,
        eps=eps,
        dtype=x.dtype,
        interpret=interpret,
        warmup=warmup,
        rep=rep,
        verbose=verbose,
    )
    return rms_norm_pallas(x, w, eps=eps, block_rows=block_rows, interpret=interpret)
