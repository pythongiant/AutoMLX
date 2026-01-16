# kernel/execute_ref.py
"""
Reference executor for KernelIR.

This is:
- Slow
- Simple
- Correct
- Deterministic

It exists purely for validation and testing.
"""

from typing import Dict
import mlx.core as mx

from kernel.ir import KernelIR, KernelOp


def _resolve_op(op: KernelOp):
    """
    Resolve a KernelOp to an MLX callable.
    """
    try:
        return getattr(mx, op.op)
    except AttributeError:
        raise NotImplementedError(f"Unsupported op in KernelIR: {op.op}")


def execute_kernel_ref(
    kernel: KernelIR,
    inputs: Dict[int, mx.array],
) -> Dict[int, mx.array]:
    """
    Execute KernelIR in a simple, correct manner.

    Args:
        kernel: KernelIR
        inputs: map tid -> mx.array

    Returns:
        env: map tid -> mx.array
    """

    # Allocate dense environment (SSA-style, tids are compact)
    max_tid = max(
        max(kernel.inputs, default=0),
        max(kernel.outputs, default=0),
        max(
            (tid for op in kernel.ops for tid in (*op.inputs, *op.outputs)),
            default=0,
        ),
    )
    env = [None] * (max_tid + 1)

    # Bind inputs
    for tid, value in inputs.items():
        env[tid] = value

    # Execute ops sequentially
    for op in kernel.ops:
        fn = _resolve_op(op)

        args = [env[tid] for tid in op.inputs]
        result = fn(*args, **(op.attrs or {}))

        # Handle multi-output ops
        if len(op.outputs) == 1:
            env[op.outputs[0]] = result
        else:
            assert isinstance(
                result, (tuple, list)
            ), f"Op {op.op} produced multiple outputs but returned {type(result)}"
            assert len(result) == len(op.outputs)
            for tid, val in zip(op.outputs, result):
                env[tid] = val

    # Return only visible tensors
    return {tid: env[tid] for tid in kernel.inputs + kernel.outputs}
