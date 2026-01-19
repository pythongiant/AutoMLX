# kernel/static_analysis.py
"""
Static analysis for rejecting unsafe generated kernels.

This module performs AST-based analysis to detect and reject kernels that:
1. Allocate memory (via creation ops like zeros, ones, arange, etc.)
2. Contain control flow branches (if/while/for)
3. Call high-level MLX ops (GEMM, convolutions, sorting, etc.)

Design:
- Analysis is purely static (no execution required)
- Conservative: when in doubt, reject
- Fast: suitable for filtering many candidates
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import List, Set, Tuple
from enum import Enum, auto


class Violation(Enum):
    """Types of violations that can be detected."""
    ALLOCATION = auto()         # Memory allocation ops
    CONTROL_FLOW = auto()       # Branching (if/while/for)
    HIGH_LEVEL_OP = auto()      # GEMM, conv, sort, etc.
    FORBIDDEN_IMPORT = auto()   # Non-mlx imports
    SIDE_EFFECT = auto()        # print, file ops, etc.
    DYNAMIC_DISPATCH = auto()   # getattr, eval, exec


# -----------------------------------------------------------------------------
# Forbidden operations (conservative lists)
# -----------------------------------------------------------------------------

# Ops that allocate new tensors (creation ops)
ALLOCATION_OPS: Set[str] = {
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "empty",
    "eye",
    "identity",
    "arange",
    "linspace",
    "logspace",
    "meshgrid",
    "diag",
    "diagonal",
    "tri",
    "tril",
    "triu",
    "array",
    "asarray",
    "from_numpy",
    "from_dlpack",
}

# High-level ops that should not appear in fused kernels
HIGH_LEVEL_OPS: Set[str] = {
    # GEMM / contractions
    "matmul",
    "addmm",
    "inner",
    "outer",
    "tensordot",
    "einsum",
    "quantized_matmul",
    "block_masked_mm",
    "gather_mm",
    "gather_qmm",
    # Convolutions
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "conv_general",
    # Sorting / selection (complex algorithms)
    "sort",
    "argsort",
    "partition",
    "argpartition",
    "topk",
    # FFT (complex algorithms)
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    # Linear algebra (potentially complex)
    "svd",
    "eigh",
    "eigvalsh",
    "qr",
    "cholesky",
    "inv",
    "pinv",
    "solve",
    "lstsq",
    # Neural network layers (stateful)
    "Linear",
    "Conv1d",
    "Conv2d",
    "LayerNorm",
    "BatchNorm",
    "Embedding",
}

# Side effect operations
SIDE_EFFECT_OPS: Set[str] = {
    "print",
    "open",
    "write",
    "save",
    "savez",
    "load",
    "eval",
    "exec",
    "compile",
    "__import__",
}

# Allowed imports
ALLOWED_IMPORTS: Set[str] = {
    "mlx",
    "mlx.core",
}


@dataclass
class AnalysisResult:
    """
    Result of static analysis.

    Attributes:
        safe: True if no violations detected
        violations: List of (violation_type, message, line_number) tuples
    """
    safe: bool
    violations: List[Tuple[Violation, str, int]]

    def __bool__(self) -> bool:
        return self.safe

    def summary(self) -> str:
        if self.safe:
            return "Static analysis passed"
        return f"Static analysis found {len(self.violations)} violation(s)"

    def format_violations(self) -> str:
        """Format violations for display."""
        lines = []
        for vtype, msg, lineno in self.violations:
            lines.append(f"  Line {lineno}: [{vtype.name}] {msg}")
        return "\n".join(lines)


class _KernelAnalyzer(ast.NodeVisitor):
    """
    AST visitor that detects forbidden patterns.
    """

    def __init__(self):
        self.violations: List[Tuple[Violation, str, int]] = []
        self.in_kernel_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "kernel":
            self.in_kernel_function = True
            self.generic_visit(node)
            self.in_kernel_function = False
        return node

    def visit_If(self, node: ast.If):
        if self.in_kernel_function:
            self.violations.append((
                Violation.CONTROL_FLOW,
                "If statement detected (branching not allowed)",
                node.lineno,
            ))
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        if self.in_kernel_function:
            self.violations.append((
                Violation.CONTROL_FLOW,
                "While loop detected (looping not allowed)",
                node.lineno,
            ))
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self.in_kernel_function:
            self.violations.append((
                Violation.CONTROL_FLOW,
                "For loop detected (looping not allowed)",
                node.lineno,
            ))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                self.violations.append((
                    Violation.FORBIDDEN_IMPORT,
                    f"Forbidden import: {alias.name}",
                    node.lineno,
                ))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module not in ALLOWED_IMPORTS:
            # Allow 'from mlx.core import ...' and 'from mlx import ...'
            if not (node.module.startswith("mlx")):
                self.violations.append((
                    Violation.FORBIDDEN_IMPORT,
                    f"Forbidden import from: {node.module}",
                    node.lineno,
                ))

    def visit_Call(self, node: ast.Call):
        if self.in_kernel_function:
            # Check for mx.op() style calls
            if isinstance(node.func, ast.Attribute):
                # e.g., mx.zeros(...)
                op_name = node.func.attr

                if op_name in ALLOCATION_OPS:
                    self.violations.append((
                        Violation.ALLOCATION,
                        f"Allocation op detected: {op_name}",
                        node.lineno,
                    ))
                elif op_name in HIGH_LEVEL_OPS:
                    self.violations.append((
                        Violation.HIGH_LEVEL_OP,
                        f"High-level op detected: {op_name}",
                        node.lineno,
                    ))
                elif op_name in SIDE_EFFECT_OPS:
                    self.violations.append((
                        Violation.SIDE_EFFECT,
                        f"Side effect op detected: {op_name}",
                        node.lineno,
                    ))

            # Check for direct function calls
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

                if func_name in SIDE_EFFECT_OPS:
                    self.violations.append((
                        Violation.SIDE_EFFECT,
                        f"Side effect function: {func_name}",
                        node.lineno,
                    ))
                elif func_name in {"getattr", "setattr", "delattr"}:
                    self.violations.append((
                        Violation.DYNAMIC_DISPATCH,
                        f"Dynamic dispatch: {func_name}",
                        node.lineno,
                    ))
                elif func_name in {"eval", "exec"}:
                    self.violations.append((
                        Violation.SIDE_EFFECT,
                        f"Code execution: {func_name}",
                        node.lineno,
                    ))

        self.generic_visit(node)


def analyze_kernel_source(source: str) -> AnalysisResult:
    """
    Analyze kernel source code for forbidden patterns.

    Args:
        source: Python source code to analyze

    Returns:
        AnalysisResult with safety status and any violations

    Example:
        >>> result = analyze_kernel_source(kernel_source)
        >>> if not result.safe:
        ...     print(result.format_violations())
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return AnalysisResult(
            safe=False,
            violations=[(Violation.SIDE_EFFECT, f"Syntax error: {e}", 0)],
        )

    analyzer = _KernelAnalyzer()
    analyzer.visit(tree)

    return AnalysisResult(
        safe=len(analyzer.violations) == 0,
        violations=analyzer.violations,
    )


def is_kernel_safe(source: str) -> bool:
    """
    Quick check if a kernel source is safe.

    Use for simple pass/fail filtering.

    Args:
        source: Kernel source code

    Returns:
        True if kernel passes all static checks
    """
    result = analyze_kernel_source(source)
    return result.safe


def reject_unsafe_kernel(source: str) -> None:
    """
    Raise an exception if kernel is unsafe.

    Use as a guard before executing generated code.

    Args:
        source: Kernel source code

    Raises:
        ValueError: If kernel contains unsafe patterns
    """
    result = analyze_kernel_source(source)
    if not result.safe:
        msg = f"Kernel rejected due to safety violations:\n{result.format_violations()}"
        raise ValueError(msg)


def filter_safe_candidates(
    candidates: List[Tuple[str, any]],
) -> List[Tuple[str, any]]:
    """
    Filter a list of candidate (source, metadata) pairs to only safe ones.

    Args:
        candidates: List of (source_code, metadata) tuples

    Returns:
        Filtered list containing only safe candidates
    """
    return [
        (source, meta) for source, meta in candidates
        if is_kernel_safe(source)
    ]
