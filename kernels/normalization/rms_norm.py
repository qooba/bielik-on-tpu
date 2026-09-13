import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def rms_norm_ref(x: jnp.ndarray, w: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x_f32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms
    return (x_norm * w).astype(x.dtype)


def _rms_norm_kernel(x_ref, w_ref, o_ref, *, eps):
    x = x_ref[...]
    w = w_ref[...]
    x_f32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms
    o_ref[...] = (x_norm * w).astype(x.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "block_rows", "interpret"))
def rms_norm_pallas(
    x: jnp.ndarray,
    w: jnp.ndarray,
    eps: float = 1e-6,
    block_rows: int = 8,
    interpret: bool = False,
) -> jnp.ndarray:
    n_rows, hidden_size = x.shape

    kernel = functools.partial(_rms_norm_kernel, eps=eps)
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(n_rows, block_rows),),
        in_specs=[
            pl.BlockSpec((block_rows, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((hidden_size,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((block_rows, hidden_size), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, w)
