import jax
import jax.numpy as jnp
import numpy as np

from kernels.matmul.matmul import matmul_pallas, matmul_ref

MAX_DIFF = 1e-2

SHAPES = [
    (512, 1536, 256, 128, 128, 128, False),
    (128, 1536, 1536, 128, 128, 128, False),
    (1, 1536, 1536, 128, 128, 128, False),
    (200, 100, 50, 128, 128, 128, False),
    (300, 300, 300, 64, 128, 128, False),
    (10, 7, 5, 128, 128, 128, False),
    (128, 1536, 1536, 128, 128, 128, True),
    (200, 100, 50, 128, 128, 128, True),
]


def _check(M, K, N, block_m, block_n, block_k, use_bias, interpret, seed=0):
    key_a, key_b, key_bias = jax.random.split(jax.random.PRNGKey(seed), 3)
    a = jax.random.normal(key_a, (M, K), dtype=jnp.float32)
    b = jax.random.normal(key_b, (K, N), dtype=jnp.float32)
    bias = jax.random.normal(key_bias, (N,), dtype=jnp.float32) if use_bias else None

    y_ref = matmul_ref(a, b, bias)
    y_np = np.array(a) @ np.array(b) + (np.array(bias) if use_bias else 0.0)
    y_pallas = matmul_pallas(
        a, b, bias, block_m=block_m, block_n=block_n, block_k=block_k, interpret=interpret
    )

    diff_ref_vs_np = np.abs(np.array(y_ref) - y_np).max()
    diff_pallas_vs_ref = float(jnp.abs(y_pallas - y_ref).max())
    diff_pallas_vs_np = np.abs(np.array(y_pallas) - y_np).max()
    has_nan = bool(jnp.isnan(y_pallas).any())

    mode = "interpret" if interpret else "real TPU"
    print(
        f"  M={M:<5} K={K:<5} N={N:<5} block=({block_m},{block_n},{block_k}) bias={use_bias!s:<5} "
        f"[{mode}] jax_vs_np={diff_ref_vs_np:.2e} pallas_vs_jax={diff_pallas_vs_ref:.2e} "
        f"pallas_vs_np={diff_pallas_vs_np:.2e} has_nan={has_nan}"
    )
    assert not has_nan, "unmasked out-of-bounds K read leaked into the output"
    assert diff_ref_vs_np < MAX_DIFF, "jax reference diverged from numpy reference"
    assert diff_pallas_vs_ref < MAX_DIFF, "pallas kernel diverged from jax reference"
    assert diff_pallas_vs_np < MAX_DIFF, "pallas kernel diverged from numpy reference"


def test_bf16_smoke(interpret):
    M, K, N = 128, 256, 128
    key_a, key_b, key_bias = jax.random.split(jax.random.PRNGKey(0), 3)
    a = jax.random.normal(key_a, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (K, N), dtype=jnp.bfloat16)
    bias = jax.random.normal(key_bias, (N,), dtype=jnp.bfloat16)

    y_ref = matmul_ref(a, b, bias)
    y_pallas = matmul_pallas(a, b, bias, interpret=interpret)

    has_nan = bool(jnp.isnan(y_pallas).any())
    scale = float(jnp.abs(y_ref.astype(jnp.float32)).max()) + 1e-6
    diff = float(jnp.abs(y_pallas.astype(jnp.float32) - y_ref.astype(jnp.float32)).max()) / scale

    mode = "interpret" if interpret else "real TPU"
    print(f"  bf16 smoke: M={M} K={K} N={N} [{mode}] rel_diff={diff:.2e} has_nan={has_nan}")
    assert not has_nan, "bf16 matmul_pallas produced NaN (Mosaic compile regression?)"
    assert diff < 5e-2, "bf16 pallas kernel diverged from bf16 jax reference"


def test_interpret_mode():
    print("interpret=True (Pallas interpreter, CPU, no Mosaic):")
    for shape in SHAPES:
        _check(*shape, interpret=True)
    test_bf16_smoke(interpret=True)


def test_real_tpu_if_available():
    has_tpu = any("TPU" in d.device_kind for d in jax.devices())
    if not has_tpu:
        print("no TPU device visible -- skipping real-TPU pass")
        return

    print("interpret=False (real TPU, Mosaic-compiled):")
    for shape in SHAPES:
        _check(*shape, interpret=False)
    test_bf16_smoke(interpret=False)


if __name__ == "__main__":
    test_interpret_mode()
    test_real_tpu_if_available()
    print("OK")
