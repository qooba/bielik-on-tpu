import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernels.common.precision import dot_precision


def matmul_ref(a: jnp.ndarray, b: jnp.ndarray, bias: jnp.ndarray | None = None) -> jnp.ndarray:
    out = jnp.dot(a, b, preferred_element_type=jnp.float32, precision=dot_precision(a.dtype))
    if bias is not None:
        out = out + bias.astype(jnp.float32)
    return out.astype(a.dtype)


def _matmul_accumulate(a_ref, b_ref, acc_ref, *, block_k, K):
    @pl.when(pl.program_id(2) == 0)
    def _zero_acc():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    if K % block_k == 0:
        a = a_ref[...]
        b = b_ref[...]
    else:
        k_remaining = K - pl.program_id(2) * block_k
        a_valid = jax.lax.broadcasted_iota(jnp.int32, (1, block_k), 1) < k_remaining
        b_valid = jax.lax.broadcasted_iota(jnp.int32, (block_k, 1), 0) < k_remaining
        a = jnp.where(a_valid, a_ref[...], 0.0)
        b = jnp.where(b_valid, b_ref[...], 0.0)
    acc_ref[...] += jnp.dot(
        a, b, preferred_element_type=jnp.float32, precision=dot_precision(a_ref.dtype)
    )


def _matmul_kernel(a_ref, b_ref, o_ref, acc_ref, *, k_steps, block_k, K):
    _matmul_accumulate(a_ref, b_ref, acc_ref, block_k=block_k, K=K)

    @pl.when(pl.program_id(2) == k_steps - 1)
    def _write_out():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _matmul_bias_kernel(a_ref, b_ref, bias_ref, o_ref, acc_ref, *, k_steps, block_k, K):
    _matmul_accumulate(a_ref, b_ref, acc_ref, block_k=block_k, K=K)

    @pl.when(pl.program_id(2) == k_steps - 1)
    def _write_out():
        o_ref[...] = (acc_ref[...] + bias_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_m", "block_n", "block_k", "interpret"))
def matmul_pallas(
    a: jnp.ndarray,
    b: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    interpret: bool = False,
) -> jnp.ndarray:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dimensions must match: {K} != {K2}"

    k_steps = pl.cdiv(K, block_k)
    grid = (pl.cdiv(M, block_m), pl.cdiv(N, block_n), k_steps)
    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda i, j, k: (i, k)),
        pl.BlockSpec((block_k, block_n), lambda i, j, k: (k, j)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j))
    out_shape = jax.ShapeDtypeStruct((M, N), a.dtype)
    scratch_shapes = [pltpu.VMEM((block_m, block_n), jnp.float32)]

    if bias is None:
        kernel = functools.partial(_matmul_kernel, k_steps=k_steps, block_k=block_k, K=K)
        return pl.pallas_call(
            kernel,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shape,
            scratch_shapes=scratch_shapes,
            interpret=interpret,
        )(a, b)

    assert bias.shape == (N,), f"bias must be ({N},), got {bias.shape}"
    kernel = functools.partial(_matmul_bias_kernel, k_steps=k_steps, block_k=block_k, K=K)
    bias_2d = bias.reshape(1, N)
    bias_spec = pl.BlockSpec((1, block_n), lambda i, j, k: (0, j))
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[*in_specs, bias_spec],
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        interpret=interpret,
    )(a, b, bias_2d)
