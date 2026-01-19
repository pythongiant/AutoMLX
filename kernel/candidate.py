# kernel/candidate.py
"""
Candidate kernel generation for a single fusion region.

This module generates multiple kernel implementations (candidates) for a given KernelIR.
Each candidate is a callable that can be evaluated for correctness and performance.

Design:
- Each candidate is generated from the same KernelIR specification
- Candidates may vary in optimization strategy or implementation approach
- All candidates must satisfy the same ABI contract (inputs/outputs/shapes/dtypes)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Dict
from enum import Enum, auto

from kernel.ir import KernelIR


class CandidateStrategy(Enum):
    """Strategy used to generate a kernel candidate."""
    REFERENCE = auto()      # Direct translation of KernelIR ops
    FUSED = auto()          # Attempt to fuse operations
    VECTORIZED = auto()     # Explicit vectorization hints
    UNROLLED = auto()       # Loop unrolling for small shapes


@dataclass
class KernelCandidate:
    """
    A single kernel candidate.

    Attributes:
        fn: The callable kernel function
        source: The source code (if available)
        strategy: The generation strategy used
        metadata: Additional metadata about the candidate
    """
    fn: Callable
    source: str
    strategy: CandidateStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args):
        return self.fn(*args)


def _generate_reference_source(kernel: KernelIR) -> str:
    """
    Generate Python source code that directly implements the KernelIR ops.

    This is the baseline implementation - straightforward translation
    of each op in sequence.
    """
    lines = [
        "import mlx.core as mx",
        "",
        "",
    ]

    # Build function signature
    args = ", ".join(f"t{tid}" for tid in kernel.inputs)
    lines.append(f"def kernel({args}):")

    # Add shape/dtype docstring
    lines.append('    """')
    lines.append('    Auto-generated kernel.')
    lines.append('    ')
    lines.append('    Inputs:')
    for tid in kernel.inputs:
        t = kernel.tensors[tid]
        lines.append(f'        t{tid}: shape={t.shape}, dtype={t.dtype}')
    lines.append('    ')
    lines.append('    Outputs:')
    for tid in kernel.outputs:
        t = kernel.tensors[tid]
        lines.append(f'        t{tid}: shape={t.shape}, dtype={t.dtype}')
    lines.append('    """')

    # Generate op implementations
    for op in kernel.ops:
        in_args = ", ".join(f"t{tid}" for tid in op.inputs)

        # Handle attrs
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

        # Generate the op call
        if len(op.outputs) == 1:
            out_var = f"t{op.outputs[0]}"
            lines.append(f"    {out_var} = mx.{op.op}({in_args}{attr_str})")
        else:
            out_vars = ", ".join(f"t{tid}" for tid in op.outputs)
            lines.append(f"    {out_vars} = mx.{op.op}({in_args}{attr_str})")

    # Return statement
    if len(kernel.outputs) == 1:
        lines.append(f"    return t{kernel.outputs[0]}")
    else:
        outs = ", ".join(f"t{tid}" for tid in kernel.outputs)
        lines.append(f"    return {outs}")

    return "\n".join(lines)


def _generate_fused_source(kernel: KernelIR) -> str:
    """
    Generate source code with explicit fusion hints.

    Uses mx.compile to hint at fusion opportunities.
    """
    # For now, wrap the reference implementation with mx.compile
    ref_source = _generate_reference_source(kernel)

    # Replace the function definition to add compile decorator
    lines = ref_source.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("def kernel("):
            new_lines.append("@mx.compile")
        new_lines.append(line)

    return "\n".join(new_lines)


def _compile_source(source: str) -> Callable:
    """
    Compile source code to a callable function.

    Args:
        source: Python source code containing a `kernel` function

    Returns:
        The compiled kernel function

    Raises:
        SyntaxError: If the source code has syntax errors
        NameError: If the source references undefined names
    """
    namespace = {}
    exec(source, namespace)
    return namespace["kernel"]


def generate_candidates(
    kernel: KernelIR,
    strategies: Optional[List[CandidateStrategy]] = None,
) -> List[KernelCandidate]:
    """
    Generate kernel candidates for a single fusion region.

    Args:
        kernel: The KernelIR specification for the region
        strategies: List of strategies to use. If None, uses default set.

    Returns:
        List of KernelCandidate objects, each with a callable `fn`,
        source code, and metadata.

    Example:
        >>> kernel = lower_region_to_kernel(graph, region)
        >>> candidates = generate_candidates(kernel)
        >>> for c in candidates:
        ...     result = c.fn(*inputs)
    """
    if strategies is None:
        strategies = [CandidateStrategy.REFERENCE, CandidateStrategy.FUSED]

    candidates = []

    for strategy in strategies:
        try:
            if strategy == CandidateStrategy.REFERENCE:
                source = _generate_reference_source(kernel)
            elif strategy == CandidateStrategy.FUSED:
                source = _generate_fused_source(kernel)
            elif strategy == CandidateStrategy.VECTORIZED:
                # For now, same as reference (placeholder for future)
                source = _generate_reference_source(kernel)
            elif strategy == CandidateStrategy.UNROLLED:
                # For now, same as reference (placeholder for future)
                source = _generate_reference_source(kernel)
            else:
                continue

            fn = _compile_source(source)

            candidates.append(KernelCandidate(
                fn=fn,
                source=source,
                strategy=strategy,
                metadata={
                    "kernel_name": kernel.name,
                    "num_ops": kernel.num_ops(),
                    "num_inputs": len(kernel.inputs),
                    "num_outputs": len(kernel.outputs),
                },
            ))
        except Exception as e:
            # Skip candidates that fail to compile
            # In production, would log this
            pass

    return candidates


def generate_single_candidate(kernel: KernelIR) -> KernelCandidate:
    """
    Generate a single reference candidate for a kernel.

    This is the simplest entry point for kernel generation.

    Args:
        kernel: The KernelIR specification

    Returns:
        A single KernelCandidate using the reference strategy

    Raises:
        ValueError: If candidate generation fails
    """
    source = _generate_reference_source(kernel)
    try:
        fn = _compile_source(source)
    except Exception as e:
        raise ValueError(f"Failed to generate candidate: {e}") from e

    return KernelCandidate(
        fn=fn,
        source=source,
        strategy=CandidateStrategy.REFERENCE,
        metadata={
            "kernel_name": kernel.name,
            "num_ops": kernel.num_ops(),
        },
    )
