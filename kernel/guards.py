# kernel/guards.py
"""
Shape and dtype guards for generated kernels.

This module provides:
1. Guard generation: Create input/output shape/dtype assertions
2. Guard wrapping: Wrap a kernel callable with runtime guards
3. Guard code generation: Emit guard code as source strings

Design principles:
- Guards are optional but recommended for safety
- Guards catch ABI violations early with clear error messages
- Guards can be disabled for performance-critical paths
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Optional
from functools import wraps

from kernel.ir import KernelIR


class GuardError(Exception):
    """Raised when a guard check fails."""
    pass


class ShapeGuardError(GuardError):
    """Raised when shape guard fails."""
    pass


class DtypeGuardError(GuardError):
    """Raised when dtype guard fails."""
    pass


@dataclass(frozen=True)
class TensorGuard:
    """
    Guard specification for a single tensor.

    Attributes:
        name: Variable name (e.g., "t0", "input_0")
        expected_shape: Expected shape tuple, or None to skip check
        expected_dtype: Expected dtype, or None to skip check
        is_input: True if this is an input tensor, False for output
    """
    name: str
    expected_shape: Optional[Tuple[int, ...]]
    expected_dtype: Any
    is_input: bool

    def check(self, tensor: Any) -> None:
        """
        Check if tensor satisfies this guard.

        Raises:
            ShapeGuardError: If shape doesn't match
            DtypeGuardError: If dtype doesn't match
        """
        role = "input" if self.is_input else "output"

        if self.expected_shape is not None:
            if tensor.shape != self.expected_shape:
                raise ShapeGuardError(
                    f"{role.capitalize()} {self.name} shape mismatch: "
                    f"expected {self.expected_shape}, got {tensor.shape}"
                )

        if self.expected_dtype is not None:
            if tensor.dtype != self.expected_dtype:
                raise DtypeGuardError(
                    f"{role.capitalize()} {self.name} dtype mismatch: "
                    f"expected {self.expected_dtype}, got {tensor.dtype}"
                )


def extract_guards(kernel: KernelIR) -> Tuple[List[TensorGuard], List[TensorGuard]]:
    """
    Extract input and output guards from a KernelIR.

    Args:
        kernel: The KernelIR specification

    Returns:
        Tuple of (input_guards, output_guards)
    """
    input_guards = []
    for tid in kernel.inputs:
        t = kernel.tensors[tid]
        input_guards.append(TensorGuard(
            name=f"t{tid}",
            expected_shape=t.shape,
            expected_dtype=t.dtype,
            is_input=True,
        ))

    output_guards = []
    for tid in kernel.outputs:
        t = kernel.tensors[tid]
        output_guards.append(TensorGuard(
            name=f"t{tid}",
            expected_shape=t.shape,
            expected_dtype=t.dtype,
            is_input=False,
        ))

    return input_guards, output_guards


def wrap_with_guards(
    fn: Callable,
    kernel: KernelIR,
    check_inputs: bool = True,
    check_outputs: bool = True,
) -> Callable:
    """
    Wrap a kernel callable with runtime shape/dtype guards.

    Args:
        fn: The kernel callable to wrap
        kernel: The KernelIR specification for guard extraction
        check_inputs: Whether to check input shapes/dtypes
        check_outputs: Whether to check output shapes/dtypes

    Returns:
        A wrapped callable that validates inputs/outputs

    Example:
        >>> guarded_fn = wrap_with_guards(kernel_fn, kernel_ir)
        >>> result = guarded_fn(input_tensor)  # Will raise if shapes don't match
    """
    input_guards, output_guards = extract_guards(kernel)

    @wraps(fn)
    def guarded_kernel(*args):
        # Check inputs
        if check_inputs:
            if len(args) != len(input_guards):
                raise GuardError(
                    f"Expected {len(input_guards)} inputs, got {len(args)}"
                )
            for guard, arg in zip(input_guards, args):
                guard.check(arg)

        # Execute kernel
        result = fn(*args)

        # Check outputs
        if check_outputs:
            # Normalize result to tuple
            if len(output_guards) == 1:
                outputs = (result,)
            else:
                outputs = result

            if len(outputs) != len(output_guards):
                raise GuardError(
                    f"Expected {len(output_guards)} outputs, got {len(outputs)}"
                )
            for guard, out in zip(output_guards, outputs):
                guard.check(out)

        return result

    return guarded_kernel


def generate_guard_source(kernel: KernelIR, indent: str = "    ") -> str:
    """
    Generate Python source code for input guards.

    This can be inserted at the beginning of a kernel function
    to add runtime validation.

    Args:
        kernel: The KernelIR specification
        indent: Indentation string to use

    Returns:
        Python source code for guard checks
    """
    lines = []
    lines.append(f"{indent}# Shape/dtype guards")

    for i, tid in enumerate(kernel.inputs):
        t = kernel.tensors[tid]
        var_name = f"t{tid}"

        if t.shape is not None:
            lines.append(
                f'{indent}assert {var_name}.shape == {t.shape!r}, '
                f'f"Input {var_name} shape mismatch: expected {t.shape!r}, got {{{var_name}.shape}}"'
            )

        if t.dtype is not None:
            # dtype comparison needs to handle MLX dtype objects
            dtype_str = str(t.dtype) if t.dtype else "None"
            lines.append(
                f'{indent}assert str({var_name}.dtype) == "{dtype_str}", '
                f'f"Input {var_name} dtype mismatch: expected {dtype_str}, got {{{var_name}.dtype}}"'
            )

    return "\n".join(lines)


def generate_guarded_kernel_source(kernel: KernelIR) -> str:
    """
    Generate complete kernel source code with embedded guards.

    This produces a self-contained kernel function that validates
    its inputs before execution.

    Args:
        kernel: The KernelIR specification

    Returns:
        Complete Python source code for a guarded kernel
    """
    lines = [
        "import mlx.core as mx",
        "",
        "",
    ]

    # Function signature
    args = ", ".join(f"t{tid}" for tid in kernel.inputs)
    lines.append(f"def kernel({args}):")

    # Docstring
    lines.append('    """')
    lines.append('    Auto-generated kernel with shape/dtype guards.')
    lines.append('    """')

    # Input guards
    for tid in kernel.inputs:
        t = kernel.tensors[tid]
        var_name = f"t{tid}"

        if t.shape is not None:
            lines.append(
                f'    assert {var_name}.shape == {t.shape!r}, '
                f'f"Input {var_name} shape mismatch: expected {t.shape!r}, got {{{var_name}.shape}}"'
            )

        if t.dtype is not None:
            dtype_str = str(t.dtype)
            lines.append(
                f'    assert str({var_name}.dtype) == "{dtype_str}", '
                f'f"Input {var_name} dtype mismatch: expected {dtype_str}, got {{{var_name}.dtype}}"'
            )

    lines.append("")

    # Operations
    for op in kernel.ops:
        in_args = ", ".join(f"t{tid}" for tid in op.inputs)

        if op.attrs:
            attr_strs = []
            for k, v in sorted(op.attrs.items()):
                if isinstance(v, str):
                    attr_strs.append(f'{k}="{v}"')
                else:
                    attr_strs.append(f'{k}={v!r}')
            attr_str = ", " + ", ".join(attr_strs) if attr_strs else ""
        else:
            attr_str = ""

        if len(op.outputs) == 1:
            out_var = f"t{op.outputs[0]}"
            lines.append(f"    {out_var} = mx.{op.op}({in_args}{attr_str})")
        else:
            out_vars = ", ".join(f"t{tid}" for tid in op.outputs)
            lines.append(f"    {out_vars} = mx.{op.op}({in_args}{attr_str})")

    # Return
    if len(kernel.outputs) == 1:
        lines.append(f"    return t{kernel.outputs[0]}")
    else:
        outs = ", ".join(f"t{tid}" for tid in kernel.outputs)
        lines.append(f"    return {outs}")

    return "\n".join(lines)
