# validate/test_kernel_backend.py
"""
Tests for KernelIR lowering, validation, reference execution, and MLX codegen.

Semantic contract:
- KernelIR represents a *single fusion region*, not the full graph
- Correctness is defined relative to executing the same region ops eagerly
"""

import pytest
import mlx.core as mx

from compile.pipeline import trace_and_compile

from kernel.lower import (
    lower_region_to_kernel,
    KernelIR,
    KernelOp,
    KernelTensor,
)
from kernel.validate import validate_kernel
from kernel.execute_ref import execute_kernel_ref
from kernel.codegen_mlx import codegen_mlx


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _map_graph_inputs_to_values(g, values):
    """
    g.inputs is a list of graph input tids in deterministic order.
    `values` is a list/tuple of mx arrays supplied to trace_and_compile.
    Return dict tid->array.
    """
    assert len(g.inputs) == len(values)
    return {tid: val for tid, val in zip(g.inputs, values)}


def _execute_region_eager(g, region, input_env):
    """
    Execute exactly the ops in `region` eagerly using MLX.

    input_env: dict tid -> mx.array
    Returns: dict tid -> mx.array (updated env)
    """
    env = dict(input_env)

    for idx in region.op_indices:
        op = g.ops[idx]
        fn = getattr(mx, op.op)
        args = [env[tid] for tid in op.inputs]
        res = fn(*args, **(op.attrs or {}))

        if len(op.outputs) == 1:
            env[op.outputs[0]] = res
        else:
            for tid, v in zip(op.outputs, res):
                env[tid] = v

    return env


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_lower_validate_execute_matches_region_semantics_single_input():
    def fn(x):
        return mx.exp(mx.sqrt(mx.abs(x) + 1))

    x = mx.random.normal((8, 8))
    g = trace_and_compile(fn, [x])

    region = g.regions[0]
    kernel = lower_region_to_kernel(g, region)
    validate_kernel(kernel)

    graph_env = _map_graph_inputs_to_values(g, [x])

    # reference kernel execution
    env_ref = execute_kernel_ref(
        kernel,
        {tid: graph_env[tid] for tid in kernel.inputs},
    )

    # eager execution of *the same region*
    env_eager = _execute_region_eager(g, region, graph_env)

    for tid in kernel.outputs:
        assert mx.allclose(env_ref[tid], env_eager[tid])

    # MLX codegen must match reference
    kernel_fn = codegen_mlx(kernel)
    args = tuple(graph_env[tid] for tid in kernel.inputs)
    cg_out = kernel_fn(*args)

    if len(kernel.outputs) == 1:
        assert mx.allclose(cg_out, env_ref[kernel.outputs[0]])
    else:
        for got, tid in zip(cg_out, kernel.outputs):
            assert mx.allclose(got, env_ref[tid])


def test_multi_output_kernel_lower_and_exec_meshgrid():
    def fn(a, b):
        X, Y = mx.meshgrid(a, b)
        return X, Y

    a = mx.arange(3)
    b = mx.arange(4)
    g = trace_and_compile(fn, [a, b])

    region = None
    for r in g.regions:
        for idx in r.op_indices:
            if g.ops[idx].op == "meshgrid":
                region = r
                break
        if region:
            break

    assert region is not None, "meshgrid region not found"

    kernel = lower_region_to_kernel(g, region)
    validate_kernel(kernel)

    graph_env = _map_graph_inputs_to_values(g, [a, b])

    env_ref = execute_kernel_ref(
        kernel,
        {tid: graph_env[tid] for tid in kernel.inputs},
    )

    env_eager = _execute_region_eager(g, region, graph_env)

    for tid in kernel.outputs:
        assert mx.allclose(env_ref[tid], env_eager[tid])

    kernel_fn = codegen_mlx(kernel)
    args = tuple(graph_env[tid] for tid in kernel.inputs)
    cg_out = kernel_fn(*args)

    assert isinstance(cg_out, tuple)
    for got, tid in zip(cg_out, kernel.outputs):
        assert mx.allclose(got, env_ref[tid])


def test_validate_rejects_read_before_write_and_dead_temp():
    # Read-before-write
    bad_tensors = {
        1: KernelTensor(tid=1, shape=(1,), dtype=None, role="output"),
    }
    bad_ops = [
        KernelOp(op="add", inputs=[999, 999], outputs=[1], attrs={}),
    ]
    bad_kernel = KernelIR(
        inputs=[],
        outputs=[1],
        temps=[],
        tensors=bad_tensors,
        ops=bad_ops,
    )

    with pytest.raises(AssertionError):
        validate_kernel(bad_kernel)

    # Dead temp
    dead_tensors = {
        10: KernelTensor(tid=10, shape=(4,), dtype=None, role="temp"),
        1: KernelTensor(tid=1, shape=(4,), dtype=None, role="output"),
    }
    ops = [
        KernelOp(op="zeros_like", inputs=[1], outputs=[1], attrs={}),
    ]
    dead_kernel = KernelIR(
        inputs=[1],
        outputs=[1],
        temps=[10],
        tensors=dead_tensors,
        ops=ops,
    )

    with pytest.raises(AssertionError):
        validate_kernel(dead_kernel)


def test_codegen_and_ref_exec_agree_on_multi_input_chain():
    def fn(x, y):
        z = x * y
        return mx.exp(z)

    x = mx.random.normal((4, 4))
    y = mx.random.normal((4, 4))
    g = trace_and_compile(fn, [x, y])

    region = g.regions[0]
    kernel = lower_region_to_kernel(g, region)
    validate_kernel(kernel)

    graph_env = _map_graph_inputs_to_values(g, [x, y])

    env_ref = execute_kernel_ref(
        kernel,
        {tid: graph_env[tid] for tid in kernel.inputs},
    )

    kernel_fn = codegen_mlx(kernel)
    args = tuple(graph_env[tid] for tid in kernel.inputs)
    cg_out = kernel_fn(*args)

    assert mx.allclose(cg_out, env_ref[kernel.outputs[0]])
