import jax
import jax.numpy as jnp

from kernels.common.tile_candidates import block_m_candidates
from kernels.normalization.autotune import _CACHE, autotune_rms_norm, rms_norm_pallas_autotuned
from kernels.normalization.rms_norm import rms_norm_ref

MAX_DIFF = 1e-5

SHAPES = [
    (128, 768),
    (10, 16),
    (1, 128),
]


def test_candidate_generation():
    for n_rows in (1, 5, 8, 10, 64, 100, 128, 300):
        for c in block_m_candidates(n_rows):
            assert (
                c % 8 == 0 or c == n_rows
            ), f"block_rows candidate {c} invalid for n_rows={n_rows}"
            assert c <= n_rows, f"block_rows candidate {c} exceeds n_rows={n_rows}"
    print("  candidate generation: all candidates tile-aligned -- OK")


def _check(n_rows, hidden_size, interpret, seed=0):
    key_x, key_w = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(key_x, (n_rows, hidden_size), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden_size,), dtype=jnp.float32)

    n_cache_entries_before = len(_CACHE)
    block_rows = autotune_rms_norm(n_rows, hidden_size, interpret=interpret, warmup=1, rep=2)
    assert len(_CACHE) == n_cache_entries_before + 1, "autotune_rms_norm did not populate the cache"

    key = (n_rows, hidden_size, 1e-6, jnp.float32, interpret)
    _CACHE[key] = block_rows
    same = autotune_rms_norm(n_rows, hidden_size, interpret=interpret, warmup=1, rep=2)
    assert same == block_rows, "autotune_rms_norm did not hit the cache on repeat call"

    y_ref = rms_norm_ref(x, w)
    y_autotuned = rms_norm_pallas_autotuned(x, w, interpret=interpret, warmup=1, rep=2)
    diff = float(jnp.abs(y_autotuned - y_ref).max())
    has_nan = bool(jnp.isnan(y_autotuned).any())

    mode = "interpret" if interpret else "real TPU"
    print(
        f"  n_rows={n_rows:<5} hidden_size={hidden_size:<5} [{mode}] "
        f"picked_block_rows={block_rows} diff={diff:.2e} has_nan={has_nan}"
    )
    assert not has_nan, "autotuned rms_norm produced NaN"
    assert diff < MAX_DIFF, "autotuned rms_norm diverged from jax reference"


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
