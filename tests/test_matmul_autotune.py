import jax
import jax.numpy as jnp

from kernels.matmul.autotune import (
    _CACHE,
    autotune_matmul,
    block_m_candidates,
    block_n_or_k_candidates,
    matmul_pallas_autotuned,
)
from kernels.matmul.matmul import matmul_ref

MAX_DIFF = 1e-2

SHAPES = [
    (256, 256, 256, False),
    (200, 100, 50, False),
    (10, 7, 5, False),
    (256, 256, 256, True),
]


def test_candidate_generation():
    for M in (1, 5, 8, 10, 64, 100, 128, 300, 1536):
        for c in block_m_candidates(M):
            assert c % 8 == 0 or c == M, f"block_m candidate {c} invalid for M={M}"
            assert c <= M, f"block_m candidate {c} exceeds M={M}"
    for dim in (1, 5, 50, 100, 128, 256, 300, 1536):
        for c in block_n_or_k_candidates(dim):
            assert c % 128 == 0 or c == dim, f"block_n/k candidate {c} invalid for dim={dim}"
            assert c <= dim, f"block_n/k candidate {c} exceeds dim={dim}"
    print("  candidate generation: all candidates tile-aligned -- OK")


def _check(M, K, N, use_bias, interpret, seed=0):
    key_a, key_b, key_bias = jax.random.split(jax.random.PRNGKey(seed), 3)
    a = jax.random.normal(key_a, (M, K), dtype=jnp.float32)
    b = jax.random.normal(key_b, (K, N), dtype=jnp.float32)
    bias = jax.random.normal(key_bias, (N,), dtype=jnp.float32) if use_bias else None

    n_cache_entries_before = len(_CACHE)
    block_m, block_n, block_k = autotune_matmul(
        M, K, N, bias=use_bias, interpret=interpret, warmup=1, rep=2
    )
    assert len(_CACHE) == n_cache_entries_before + 1, "autotune_matmul did not populate the cache"

    key = (M, K, N, jnp.float32, use_bias, interpret)
    _CACHE[key] = (block_m, block_n, block_k)
    same = autotune_matmul(M, K, N, bias=use_bias, interpret=interpret, warmup=1, rep=2)
    expected = (block_m, block_n, block_k)
    assert same == expected, "autotune_matmul did not hit the cache on repeat call"

    y_ref = matmul_ref(a, b, bias)
    y_autotuned = matmul_pallas_autotuned(a, b, bias, interpret=interpret, warmup=1, rep=2)
    diff = float(jnp.abs(y_autotuned - y_ref).max())
    has_nan = bool(jnp.isnan(y_autotuned).any())

    mode = "interpret" if interpret else "real TPU"
    print(
        f"  M={M:<5} K={K:<5} N={N:<5} bias={use_bias!s:<5} [{mode}] "
        f"picked_block=({block_m},{block_n},{block_k}) diff_vs_ref={diff:.2e} has_nan={has_nan}"
    )
    assert not has_nan, "autotuned matmul produced NaN"
    assert diff < MAX_DIFF, "autotuned matmul diverged from jax reference"


def test_interpret_mode():
    print("interpret=True (Pallas interpreter, CPU, no Mosaic):")
    test_candidate_generation()
    for shape in SHAPES:
        _check(*shape, interpret=True)


def test_real_tpu_if_available():
    has_tpu = any("TPU" in d.device_kind for d in jax.devices())
    if not has_tpu:
        print("no TPU device visible -- skipping real-TPU pass")
        return

    print("interpret=False (real TPU, Mosaic-compiled):")
    for shape in SHAPES:
        _check(*shape, interpret=False)


if __name__ == "__main__":
    test_interpret_mode()
    test_real_tpu_if_available()
    print("OK")
