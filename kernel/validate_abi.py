# kernel/validate_abi.py
"""
Static validation of generated kernel code against KernelIR ABI.

This module validates that a generated kernel (either as source code or callable)
conforms to the ABI contract specified by a KernelIR:
- Correct number of parameters
- Correct function signature
- Returns correct number of outputs
- Uses expected tensor variables

Validation is performed statically (via AST analysis) without execution.
"""

from __future__ import annotations
import ast
import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple
from enum import Enum, auto

from kernel.ir import KernelIR


class ABIViolation(Enum):
    """Types of ABI violations."""
    WRONG_PARAM_COUNT = auto()
    MISSING_PARAM = auto()
    EXTRA_PARAM = auto()
    WRONG_RETURN_COUNT = auto()
    MISSING_RETURN_VAR = auto()
    UNDEFINED_VAR_USED = auto()
    NO_RETURN_STATEMENT = auto()
    MULTIPLE_FUNCTIONS = auto()
    NO_FUNCTION_FOUND = auto()


@dataclass
class ABIValidationResult:
    """
    Result of ABI validation.

    Attributes:
        valid: True if the kernel passes all ABI checks
        violations: List of (violation_type, message) tuples
        warnings: List of warning messages (non-fatal issues)
    """
    valid: bool
    violations: List[Tuple[ABIViolation, str]]
    warnings: List[str]

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        """Return a human-readable summary."""
        if self.valid:
            msg = "ABI validation passed"
            if self.warnings:
                msg += f" with {len(self.warnings)} warning(s)"
            return msg

        return f"ABI validation failed with {len(self.violations)} violation(s)"


class _KernelASTVisitor(ast.NodeVisitor):
    """
    AST visitor for kernel function analysis.

    Extracts:
    - Function parameters
    - Return statements
    - Variable assignments
    - Variable references
    """

    def __init__(self):
        self.functions: List[ast.FunctionDef] = []
        self.parameters: List[str] = []
        self.return_values: List[ast.AST] = []
        self.assignments: Set[str] = set()
        self.references: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "kernel":
            self.functions.append(node)
            self.parameters = [arg.arg for arg in node.args.args]
            # Visit body
            for stmt in node.body:
                self.visit(stmt)
        return node

    def visit_Return(self, node: ast.Return):
        if node.value:
            self.return_values.append(node.value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assignments.add(elt.id)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.references.add(node.id)
        self.generic_visit(node)


def _extract_return_names(return_node: ast.AST) -> List[str]:
    """Extract variable names from a return statement."""
    if isinstance(return_node, ast.Name):
        return [return_node.id]
    elif isinstance(return_node, ast.Tuple):
        names = []
        for elt in return_node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
        return names
    return []


def validate_kernel_abi(
    source: str,
    kernel: KernelIR,
) -> ABIValidationResult:
    """
    Validate kernel source code against KernelIR ABI.

    This performs static analysis to check:
    1. Function signature matches expected parameters
    2. Return statement exists and returns expected outputs
    3. All referenced variables are defined

    Args:
        source: Python source code containing a `kernel` function
        kernel: The KernelIR specification defining the ABI

    Returns:
        ABIValidationResult with validation status and any violations

    Example:
        >>> result = validate_kernel_abi(source_code, kernel_ir)
        >>> if not result.valid:
        ...     for violation, msg in result.violations:
        ...         print(f"{violation.name}: {msg}")
    """
    violations = []
    warnings = []

    # Parse source
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ABIValidationResult(
            valid=False,
            violations=[(ABIViolation.NO_FUNCTION_FOUND, f"Syntax error: {e}")],
            warnings=[],
        )

    # Extract function info
    visitor = _KernelASTVisitor()
    visitor.visit(tree)

    # Check for kernel function
    if not visitor.functions:
        return ABIValidationResult(
            valid=False,
            violations=[(ABIViolation.NO_FUNCTION_FOUND, "No 'kernel' function found")],
            warnings=[],
        )

    if len(visitor.functions) > 1:
        warnings.append("Multiple 'kernel' functions found, using first one")

    # Expected parameters
    expected_params = [f"t{tid}" for tid in kernel.inputs]

    # Check parameter count
    if len(visitor.parameters) != len(expected_params):
        violations.append((
            ABIViolation.WRONG_PARAM_COUNT,
            f"Expected {len(expected_params)} parameters, found {len(visitor.parameters)}"
        ))

    # Check parameter names
    for expected in expected_params:
        if expected not in visitor.parameters:
            violations.append((
                ABIViolation.MISSING_PARAM,
                f"Missing expected parameter: {expected}"
            ))

    for param in visitor.parameters:
        if param not in expected_params:
            violations.append((
                ABIViolation.EXTRA_PARAM,
                f"Unexpected parameter: {param}"
            ))

    # Check return statement
    if not visitor.return_values:
        violations.append((
            ABIViolation.NO_RETURN_STATEMENT,
            "No return statement found in kernel function"
        ))
    else:
        # Get the last return statement (most relevant)
        last_return = visitor.return_values[-1]
        return_names = _extract_return_names(last_return)

        # Expected return variables
        expected_returns = [f"t{tid}" for tid in kernel.outputs]

        if len(return_names) != len(expected_returns):
            violations.append((
                ABIViolation.WRONG_RETURN_COUNT,
                f"Expected {len(expected_returns)} return values, found {len(return_names)}"
            ))

        for expected in expected_returns:
            if expected not in return_names:
                violations.append((
                    ABIViolation.MISSING_RETURN_VAR,
                    f"Missing expected return variable: {expected}"
                ))

    # Check for undefined variable references
    defined = set(visitor.parameters) | visitor.assignments
    # Add common builtins and mlx namespace
    defined.add("mx")
    defined.add("mlx")

    undefined = visitor.references - defined
    # Filter out attribute access (e.g., mx.add is not undefined)
    for var in undefined:
        if var not in {"mx", "mlx", "True", "False", "None"}:
            # This might be a false positive if it's an attribute
            # For now, just warn
            warnings.append(f"Potentially undefined variable: {var}")

    return ABIValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def validate_kernel_callable(
    fn: Callable,
    kernel: KernelIR,
) -> ABIValidationResult:
    """
    Validate a kernel callable against KernelIR ABI.

    Uses inspect to extract function signature.

    Args:
        fn: The kernel callable to validate
        kernel: The KernelIR specification

    Returns:
        ABIValidationResult with validation status
    """
    violations = []
    warnings = []

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError) as e:
        return ABIValidationResult(
            valid=False,
            violations=[(ABIViolation.NO_FUNCTION_FOUND, f"Cannot inspect function: {e}")],
            warnings=[],
        )

    # Check parameter count
    params = list(sig.parameters.keys())
    expected_count = len(kernel.inputs)

    if len(params) != expected_count:
        violations.append((
            ABIViolation.WRONG_PARAM_COUNT,
            f"Expected {expected_count} parameters, found {len(params)}"
        ))

    # Try to get source for deeper analysis
    try:
        source = inspect.getsource(fn)
        source_result = validate_kernel_abi(source, kernel)
        violations.extend(source_result.violations)
        warnings.extend(source_result.warnings)
    except (OSError, TypeError):
        # Can't get source, skip source-level validation
        warnings.append("Could not retrieve source code for full validation")

    return ABIValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def validate_abi_strict(source: str, kernel: KernelIR) -> bool:
    """
    Strict ABI validation that returns a simple bool.

    Use this for quick checks where you just need pass/fail.

    Args:
        source: Kernel source code
        kernel: KernelIR specification

    Returns:
        True if ABI is valid, False otherwise
    """
    result = validate_kernel_abi(source, kernel)
    return result.valid
