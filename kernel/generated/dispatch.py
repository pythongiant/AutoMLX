# kernel/generated/dispatch.py
"""
Generated kernel dispatch logic.

This is the ONLY place where generated kernels are executed.

Rules:
- Generated kernels are always optional
- Any failure falls back to reference execution
- No exceptions escape this layer
"""

from typing import Tuple, Any

import mlx.core as mx

from kernel.ir import KernelIR
from kernel.execute_ref import execute_kernel_ref
from kernel.generated.cache import get_kernel, remove_kernel
from kernel.generated.registry import generated_kernels_enabled
from kernel.generated.cache import kernel_signature, get_kernel
from kernel.execute_ref import execute_kernel_ref


def execute_kernel(kernel_ir, inputs):
    """
    Execute a KernelIR using:
    1. generated kernel if available
    2. reference executor otherwise
    """

    sig = kernel_signature(kernel_ir)
    fn = get_kernel(sig)

    if fn is not None:
        try:
            # positional args in kernel.inputs order
            args = [inputs[tid] for tid in kernel_ir.inputs]
            out = fn(*args)

            # Normalize output
            if len(kernel_ir.outputs) == 1:
                return {kernel_ir.outputs[0]: out}

            return {
                tid: val
                for tid, val in zip(kernel_ir.outputs, out)
            }

        except Exception:
            # Hard fallback on ANY failure
            pass

    # Safe fallback
    return execute_kernel_ref(kernel_ir, inputs)

def _validate_outputs(
    outputs: Any,
    kernel: KernelIR,
) -> bool:
    """
    Validate generated kernel outputs against KernelIR ABI.
    """
    if len(kernel.outputs) == 1:
        outputs = (outputs,)

    if not isinstance(outputs, tuple):
        return False

    if len(outputs) != len(kernel.outputs):
        return False

    for arr, tid in zip(outputs, kernel.outputs):
        expected = kernel.tensors[tid]
        if arr.shape != expected.shape:
            return False
        if arr.dtype != expected.dtype:
            return False

    return True


def execute_region(
    kernel: KernelIR,
    inputs: Tuple[mx.array, ...],
    signature: str,
):
    """
    Execute a KernelIR region.

    Order:
    1. Try generated kernel (if enabled + present)
    2. Validate outputs
    3. Fallback to reference executor on ANY failure
    """

    # ------------------------------------------------------------------
    # Try generated kernel
    # ------------------------------------------------------------------
    if generated_kernels_enabled():
        gen_kernel = get_kernel(signature)
        if gen_kernel is not None:
            try:
                out = gen_kernel(*inputs)
                if _validate_outputs(out, kernel):
                    return out
                else:
                    # permanently disable this kernel
                    remove_kernel(signature)
            except Exception:
                # any failure → disable kernel
                remove_kernel(signature)

    # ------------------------------------------------------------------
    # Fallback: reference executor (always correct)
    # ------------------------------------------------------------------
    input_env = {tid: val for tid, val in zip(kernel.inputs, inputs)}
    env = execute_kernel_ref(kernel, input_env)

    if len(kernel.outputs) == 1:
        return env[kernel.outputs[0]]

    return tuple(env[tid] for tid in kernel.outputs)
