# kernel/correctness.py
"""
Correctness checking for generated kernels against KernelIR reference executor.

This module validates that a generated kernel produces the same outputs
as the reference executor for a given set of test inputs.

Design:
- Reference executor (execute_kernel_ref) is the source of truth
- Generated kernels must produce outputs within tolerance of reference
- Multiple test cases can be run for increased confidence
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Any, Optional
import mlx.core as mx

from kernel.ir import KernelIR
from kernel.execute_ref import execute_kernel_ref


@dataclass
class CorrectnessResult:
    """
    Result of a correctness check.

    Attributes:
        passed: True if all outputs match within tolerance
        num_tests: Number of test cases run
        failures: List of (test_index, output_index, message) for failures
        max_error: Maximum error observed across all tests
    """
    passed: bool
    num_tests: int
    failures: List[Tuple[int, int, str]]
    max_error: float

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return f"Correctness check passed ({self.num_tests} tests, max_error={self.max_error:.2e})"
        return f"Correctness check failed ({len(self.failures)} failures in {self.num_tests} tests)"


def _generate_random_input(shape: tuple, dtype: Any) -> mx.array:
    """Generate a random tensor with the given shape and dtype."""
    # Handle different dtypes
    dtype_str = str(dtype) if dtype else "float32"

    if "int" in dtype_str:
        # Integer types: use random integers
        return mx.random.randint(0, 10, shape=shape).astype(dtype)
    elif "bool" in dtype_str:
        # Boolean: random binary
        return mx.random.randint(0, 2, shape=shape).astype(mx.bool_)
    else:
        # Float types: normal distribution
        arr = mx.random.normal(shape=shape)
        if dtype:
            arr = arr.astype(dtype)
        return arr


def _generate_test_inputs(kernel: KernelIR, seed: int = 42) -> Dict[int, mx.array]:
    """Generate random test inputs for a kernel."""
    mx.random.seed(seed)

    inputs = {}
    for tid in kernel.inputs:
        t = kernel.tensors[tid]
        inputs[tid] = _generate_random_input(t.shape, t.dtype)

    return inputs


def _compare_outputs(
    candidate_out: Any,
    reference_out: Dict[int, mx.array],
    kernel: KernelIR,
    rtol: float,
    atol: float,
) -> Tuple[bool, float, List[Tuple[int, str]]]:
    """
    Compare candidate outputs against reference.

    Returns:
        (all_match, max_error, failures)
    """
    # Normalize candidate output to dict
    if len(kernel.outputs) == 1:
        candidate_dict = {kernel.outputs[0]: candidate_out}
    else:
        candidate_dict = {
            tid: val for tid, val in zip(kernel.outputs, candidate_out)
        }

    failures = []
    max_error = 0.0
    all_match = True

    for i, tid in enumerate(kernel.outputs):
        expected = reference_out[tid]
        actual = candidate_dict.get(tid)

        if actual is None:
            failures.append((i, f"Output t{tid} is missing"))
            all_match = False
            continue

        # Check shape
        if actual.shape != expected.shape:
            failures.append((i, f"Output t{tid} shape mismatch: expected {expected.shape}, got {actual.shape}"))
            all_match = False
            continue

        # Check dtype
        if actual.dtype != expected.dtype:
            failures.append((i, f"Output t{tid} dtype mismatch: expected {expected.dtype}, got {actual.dtype}"))
            all_match = False
            continue

        # Check values
        if not mx.allclose(actual, expected, rtol=rtol, atol=atol).item():
            # Compute actual error
            diff = mx.abs(actual - expected)
            error = mx.max(diff).item()
            max_error = max(max_error, error)
            failures.append((i, f"Output t{tid} values differ (max_error={error:.2e})"))
            all_match = False
        else:
            # Track max error even for passing cases
            diff = mx.abs(actual - expected)
            error = mx.max(diff).item()
            max_error = max(max_error, error)

    return all_match, max_error, failures


def check_correctness(
    fn: Callable,
    kernel: KernelIR,
    num_tests: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    seeds: Optional[List[int]] = None,
) -> CorrectnessResult:
    """
    Check that a kernel callable produces correct outputs.

    Runs the kernel with random inputs and compares against the
    reference executor.

    Args:
        fn: The kernel callable to test
        kernel: The KernelIR specification
        num_tests: Number of random test cases to run
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        seeds: Optional list of seeds for reproducibility

    Returns:
        CorrectnessResult with test outcomes

    Example:
        >>> result = check_correctness(kernel_fn, kernel_ir)
        >>> if not result.passed:
        ...     print(f"Failed: {result.failures}")
    """
    if seeds is None:
        seeds = list(range(num_tests))

    all_failures = []
    max_error = 0.0

    for test_idx, seed in enumerate(seeds[:num_tests]):
        # Generate inputs
        inputs = _generate_test_inputs(kernel, seed)

        # Run reference
        ref_result = execute_kernel_ref(kernel, inputs)

        # Run candidate
        try:
            args = tuple(inputs[tid] for tid in kernel.inputs)
            candidate_result = fn(*args)
        except Exception as e:
            all_failures.append((test_idx, -1, f"Execution error: {e}"))
            continue

        # Compare
        match, error, failures = _compare_outputs(
            candidate_result, ref_result, kernel, rtol, atol
        )
        max_error = max(max_error, error)

        if not match:
            for out_idx, msg in failures:
                all_failures.append((test_idx, out_idx, msg))

    return CorrectnessResult(
        passed=len(all_failures) == 0,
        num_tests=num_tests,
        failures=all_failures,
        max_error=max_error,
    )


def check_correctness_with_inputs(
    fn: Callable,
    kernel: KernelIR,
    test_inputs: List[Dict[int, mx.array]],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> CorrectnessResult:
    """
    Check correctness with specific test inputs.

    Use this when you have predetermined inputs to test with.

    Args:
        fn: The kernel callable to test
        kernel: The KernelIR specification
        test_inputs: List of input dictionaries (tid -> array)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        CorrectnessResult
    """
    all_failures = []
    max_error = 0.0

    for test_idx, inputs in enumerate(test_inputs):
        # Run reference
        ref_result = execute_kernel_ref(kernel, inputs)

        # Run candidate
        try:
            args = tuple(inputs[tid] for tid in kernel.inputs)
            candidate_result = fn(*args)
        except Exception as e:
            all_failures.append((test_idx, -1, f"Execution error: {e}"))
            continue

        # Compare
        match, error, failures = _compare_outputs(
            candidate_result, ref_result, kernel, rtol, atol
        )
        max_error = max(max_error, error)

        if not match:
            for out_idx, msg in failures:
                all_failures.append((test_idx, out_idx, msg))

    return CorrectnessResult(
        passed=len(all_failures) == 0,
        num_tests=len(test_inputs),
        failures=all_failures,
        max_error=max_error,
    )


def verify_candidate(
    fn: Callable,
    kernel: KernelIR,
    quick: bool = True,
) -> bool:
    """
    Quick verification that a candidate kernel is correct.

    This is a convenience function for simple pass/fail checks.

    Args:
        fn: The kernel callable
        kernel: The KernelIR specification
        quick: If True, run fewer tests (faster but less thorough)

    Returns:
        True if kernel passes correctness checks
    """
    num_tests = 2 if quick else 10
    result = check_correctness(fn, kernel, num_tests=num_tests)
    return result.passed
