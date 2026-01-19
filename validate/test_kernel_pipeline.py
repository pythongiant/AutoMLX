# validate/test_kernel_pipeline.py
"""
Tests for the complete kernel generation and validation pipeline.

Tests the following modules:
- kernel/candidate.py: Candidate generation
- kernel/guards.py: Shape/dtype guards
- kernel/validate_abi.py: ABI validation
- kernel/correctness.py: Correctness checking
- kernel/static_analysis.py: Safety analysis
"""

import pytest
import mlx.core as mx

from compile.pipeline import trace_and_compile
from kernel.lower import lower_region_to_kernel
from kernel.ir import KernelIR, KernelOp, KernelTensor

# Import new modules
from kernel.candidate import (
    generate_candidates,
    generate_single_candidate,
    CandidateStrategy,
    KernelCandidate,
)
from kernel.guards import (
    wrap_with_guards,
    extract_guards,
    generate_guarded_kernel_source,
    GuardError,
    ShapeGuardError,
    DtypeGuardError,
)
from kernel.validate_abi import (
    validate_kernel_abi,
    validate_kernel_callable,
    validate_abi_strict,
    ABIViolation,
)
from kernel.correctness import (
    check_correctness,
    check_correctness_with_inputs,
    verify_candidate,
)
from kernel.static_analysis import (
    analyze_kernel_source,
    is_kernel_safe,
    reject_unsafe_kernel,
    Violation,
)


# -----------------------------------------------------------------------------
# Helper to create test kernels
# -----------------------------------------------------------------------------

def _make_simple_kernel():
    """Create a simple kernel for testing."""
    def fn(x):
        return mx.exp(mx.sqrt(mx.abs(x) + 1))

    x = mx.random.normal((8, 8))
    g = trace_and_compile(fn, [x])
    region = g.regions[0]
    kernel = lower_region_to_kernel(g, region)
    return kernel, x


def _make_multi_input_kernel():
    """Create a multi-input kernel."""
    def fn(x, y):
        return mx.add(x, y)

    x = mx.random.normal((4, 4))
    y = mx.random.normal((4, 4))
    g = trace_and_compile(fn, [x, y])
    region = g.regions[0]
    kernel = lower_region_to_kernel(g, region)
    return kernel, x, y


# -----------------------------------------------------------------------------
# Tests: Candidate Generation
# -----------------------------------------------------------------------------

class TestCandidateGeneration:

    def test_generate_single_candidate(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        assert candidate is not None
        assert candidate.fn is not None
        assert candidate.source is not None
        assert candidate.strategy == CandidateStrategy.REFERENCE

    def test_generate_multiple_candidates(self):
        kernel, x = _make_simple_kernel()
        candidates = generate_candidates(kernel)

        assert len(candidates) >= 1
        assert all(isinstance(c, KernelCandidate) for c in candidates)

    def test_candidate_executes_correctly(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        # Execute
        result = candidate.fn(x)

        # Should produce output of correct shape
        expected_shape = kernel.tensors[kernel.outputs[0]].shape
        assert result.shape == expected_shape

    def test_candidate_source_contains_kernel_function(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        assert "def kernel(" in candidate.source
        assert "import mlx.core as mx" in candidate.source


# -----------------------------------------------------------------------------
# Tests: Shape/Dtype Guards
# -----------------------------------------------------------------------------

class TestGuards:

    def test_extract_guards(self):
        kernel, x = _make_simple_kernel()
        input_guards, output_guards = extract_guards(kernel)

        assert len(input_guards) == len(kernel.inputs)
        assert len(output_guards) == len(kernel.outputs)

    def test_wrap_with_guards_passes_valid_input(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        guarded_fn = wrap_with_guards(candidate.fn, kernel)
        result = guarded_fn(x)

        assert result is not None

    def test_wrap_with_guards_rejects_wrong_shape(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        guarded_fn = wrap_with_guards(candidate.fn, kernel)

        # Wrong shape input
        wrong_shape = mx.random.normal((16, 16))

        with pytest.raises(ShapeGuardError):
            guarded_fn(wrong_shape)

    def test_wrap_with_guards_rejects_wrong_input_count(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        guarded_fn = wrap_with_guards(candidate.fn, kernel)

        with pytest.raises(GuardError):
            guarded_fn(x, x)  # Too many inputs

    def test_generate_guarded_kernel_source(self):
        kernel, x = _make_simple_kernel()
        source = generate_guarded_kernel_source(kernel)

        assert "assert" in source  # Should have guard assertions
        assert "def kernel(" in source


# -----------------------------------------------------------------------------
# Tests: ABI Validation
# -----------------------------------------------------------------------------

class TestABIValidation:

    def test_valid_kernel_passes_abi(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        result = validate_kernel_abi(candidate.source, kernel)
        assert result.valid

    def test_wrong_param_count_fails_abi(self):
        kernel, x = _make_simple_kernel()

        bad_source = """
import mlx.core as mx

def kernel(t0, t1):  # Wrong: expects 1 param, has 2
    return mx.exp(t0)
"""
        result = validate_kernel_abi(bad_source, kernel)
        assert not result.valid
        assert any(v[0] == ABIViolation.WRONG_PARAM_COUNT for v in result.violations)

    def test_missing_return_fails_abi(self):
        kernel, x = _make_simple_kernel()

        bad_source = """
import mlx.core as mx

def kernel(t0):
    x = mx.exp(t0)
    # No return!
"""
        result = validate_kernel_abi(bad_source, kernel)
        assert not result.valid
        assert any(v[0] == ABIViolation.NO_RETURN_STATEMENT for v in result.violations)

    def test_validate_abi_strict(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        assert validate_abi_strict(candidate.source, kernel)


# -----------------------------------------------------------------------------
# Tests: Correctness Checking
# -----------------------------------------------------------------------------

class TestCorrectness:

    def test_correct_kernel_passes(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        result = check_correctness(candidate.fn, kernel, num_tests=3)
        assert result.passed

    def test_verify_candidate(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        assert verify_candidate(candidate.fn, kernel)

    def test_incorrect_kernel_fails(self):
        kernel, x = _make_simple_kernel()

        # Create a wrong kernel that returns zeros
        def wrong_kernel(t0):
            return mx.zeros_like(t0)

        result = check_correctness(wrong_kernel, kernel, num_tests=3)
        assert not result.passed

    def test_check_correctness_with_specific_inputs(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        # Create specific test inputs
        test_inputs = [
            {kernel.inputs[0]: mx.ones((8, 8))},
            {kernel.inputs[0]: mx.zeros((8, 8))},
        ]

        result = check_correctness_with_inputs(
            candidate.fn, kernel, test_inputs
        )
        assert result.passed


# -----------------------------------------------------------------------------
# Tests: Static Analysis
# -----------------------------------------------------------------------------

class TestStaticAnalysis:

    def test_safe_kernel_passes(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        result = analyze_kernel_source(candidate.source)
        assert result.safe

    def test_allocation_detected(self):
        bad_source = """
import mlx.core as mx

def kernel(t0):
    temp = mx.zeros((8, 8))  # Allocation!
    return t0 + temp
"""
        result = analyze_kernel_source(bad_source)
        assert not result.safe
        assert any(v[0] == Violation.ALLOCATION for v in result.violations)

    def test_branching_detected(self):
        bad_source = """
import mlx.core as mx

def kernel(t0):
    if t0.shape[0] > 10:  # Branching!
        return mx.exp(t0)
    else:
        return mx.sqrt(t0)
"""
        result = analyze_kernel_source(bad_source)
        assert not result.safe
        assert any(v[0] == Violation.CONTROL_FLOW for v in result.violations)

    def test_high_level_op_detected(self):
        bad_source = """
import mlx.core as mx

def kernel(t0, t1):
    return mx.matmul(t0, t1)  # High-level op!
"""
        result = analyze_kernel_source(bad_source)
        assert not result.safe
        assert any(v[0] == Violation.HIGH_LEVEL_OP for v in result.violations)

    def test_loop_detected(self):
        bad_source = """
import mlx.core as mx

def kernel(t0):
    result = t0
    for i in range(10):  # Loop!
        result = mx.exp(result)
    return result
"""
        result = analyze_kernel_source(bad_source)
        assert not result.safe
        assert any(v[0] == Violation.CONTROL_FLOW for v in result.violations)

    def test_forbidden_import_detected(self):
        bad_source = """
import os  # Forbidden!
import mlx.core as mx

def kernel(t0):
    return mx.exp(t0)
"""
        result = analyze_kernel_source(bad_source)
        assert not result.safe
        assert any(v[0] == Violation.FORBIDDEN_IMPORT for v in result.violations)

    def test_is_kernel_safe_convenience(self):
        kernel, x = _make_simple_kernel()
        candidate = generate_single_candidate(kernel)

        assert is_kernel_safe(candidate.source)

    def test_reject_unsafe_kernel_raises(self):
        bad_source = """
import mlx.core as mx

def kernel(t0):
    temp = mx.zeros((8, 8))
    return t0
"""
        with pytest.raises(ValueError):
            reject_unsafe_kernel(bad_source)


# -----------------------------------------------------------------------------
# Tests: Integration (Full Pipeline)
# -----------------------------------------------------------------------------

class TestFullPipeline:

    def test_end_to_end_single_input(self):
        """Test complete pipeline: generate, validate, check correctness."""
        kernel, x = _make_simple_kernel()

        # 1. Generate candidate
        candidate = generate_single_candidate(kernel)
        assert candidate is not None

        # 2. Validate ABI
        abi_result = validate_kernel_abi(candidate.source, kernel)
        assert abi_result.valid

        # 3. Static analysis
        analysis = analyze_kernel_source(candidate.source)
        assert analysis.safe

        # 4. Correctness check
        correctness = check_correctness(candidate.fn, kernel)
        assert correctness.passed

        # 5. Wrap with guards and execute
        guarded_fn = wrap_with_guards(candidate.fn, kernel)
        result = guarded_fn(x)
        assert result.shape == x.shape

    def test_end_to_end_multi_input(self):
        """Test pipeline with multi-input kernel."""
        kernel, x, y = _make_multi_input_kernel()

        # Generate and validate
        candidate = generate_single_candidate(kernel)
        assert validate_abi_strict(candidate.source, kernel)
        assert is_kernel_safe(candidate.source)
        assert verify_candidate(candidate.fn, kernel)

        # Execute with guards
        guarded_fn = wrap_with_guards(candidate.fn, kernel)
        result = guarded_fn(x, y)
        assert result.shape == x.shape

    def test_multiple_candidates_all_correct(self):
        """Test that all generated candidates are correct."""
        kernel, x = _make_simple_kernel()
        candidates = generate_candidates(kernel)

        for candidate in candidates:
            assert validate_abi_strict(candidate.source, kernel)
            assert is_kernel_safe(candidate.source)
            assert verify_candidate(candidate.fn, kernel, quick=True)
