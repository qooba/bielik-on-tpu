import jax
import jax.numpy as jnp
import numpy as np

from kernels.normalization.rms_norm import rms_norm_pallas, rms_norm_ref

EPS = 1e-6
MAX_DIFF = 1e-5

SHAPES = [
    (128, 768, 8),
    (128, 768, 128),
    (4096, 4096, 128),
    (200, 768, 128),
    (347, 128, 128),
]


def rms_norm_numpy(x, w, eps=EPS):
    x = x.astype(np.float32)
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + eps)
    return (x / rms * w).astype(np.float32)


def _check(n_rows, hidden_size, block_rows, interpret, seed=0):
    key_x, key_w = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(key_x, (n_rows, hidden_size), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden_size,), dtype=jnp.float32)

    y_ref = rms_norm_ref(x, w, eps=EPS)
    y_np = rms_norm_numpy(np.array(x), np.array(w))
    y_pallas = rms_norm_pallas(x, w, eps=EPS, block_rows=block_rows, interpret=interpret)

    diff_ref_vs_np = np.abs(np.array(y_ref) - y_np).max()
    diff_pallas_vs_ref = float(jnp.abs(y_pallas - y_ref).max())
    diff_pallas_vs_np = np.abs(np.array(y_pallas) - y_np).max()

    mode = "interpret" if interpret else "real TPU"
    print(
        f"  n_rows={n_rows:<5} hidden_size={hidden_size:<5} block_rows={block_rows:<4} "
        f"[{mode}] jax_vs_np={diff_ref_vs_np:.2e} pallas_vs_jax={diff_pallas_vs_ref:.2e} "
        f"pallas_vs_np={diff_pallas_vs_np:.2e}"
    )
    assert diff_ref_vs_np < MAX_DIFF, "jax reference diverged from numpy reference"
    assert diff_pallas_vs_ref < MAX_DIFF, "pallas kernel diverged from jax reference"
    assert diff_pallas_vs_np < MAX_DIFF, "pallas kernel diverged from numpy reference"


def test_bf16_precision(interpret):
    n_rows, hidden_size = 128, 1536
    key_x, key_w = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key_x, (n_rows, hidden_size), dtype=jnp.bfloat16)
    w = jax.random.normal(key_w, (hidden_size,), dtype=jnp.bfloat16)

    y_ref = rms_norm_ref(x, w, eps=EPS)
    y_pallas = rms_norm_pallas(x, w, eps=EPS, block_rows=8, interpret=interpret)

    has_nan = bool(jnp.isnan(y_pallas).any())
    scale = float(jnp.abs(y_ref.astype(jnp.float32)).max()) + 1e-6
    diff = float(jnp.abs(y_pallas.astype(jnp.float32) - y_ref.astype(jnp.float32)).max()) / scale

    mode = "interpret" if interpret else "real TPU"
    print(
        f"  bf16 precision: n_rows={n_rows} hidden_size={hidden_size} [{mode}] rel_diff={diff:.2e}"
    )
    assert not has_nan, "bf16 rms_norm_pallas produced NaN"
    assert diff < 5e-3, "bf16 pallas kernel diverged from bf16 jax reference"


def test_interpret_mode():
    print("interpret=True (Pallas interpreter, CPU, no Mosaic):")
    for n_rows, hidden_size, block_rows in SHAPES:
        _check(n_rows, hidden_size, block_rows, interpret=True)
    test_bf16_precision(interpret=True)


def test_real_tpu_if_available():
    has_tpu = any("TPU" in d.device_kind for d in jax.devices())
    if not has_tpu:
        print("no TPU device visible -- skipping real-TPU pass")
        return

    print("interpret=False (real TPU, Mosaic-compiled):")
    for n_rows, hidden_size, block_rows in SHAPES:
        _check(n_rows, hidden_size, block_rows, interpret=False)
    test_bf16_precision(interpret=False)


if __name__ == "__main__":
    test_interpret_mode()
    test_real_tpu_if_available()
    print("OK")
