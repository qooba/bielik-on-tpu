import jax
import jax.numpy as jnp


def dot_precision(dtype: jnp.dtype) -> jax.lax.Precision | None:
    return jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None
