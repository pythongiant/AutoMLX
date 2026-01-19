"""
Benchmark: Softmax Operations
=============================

Canonicalization and reduction ordering benchmark.

Softmax is canonicalized into primitive ops:
  softmax(x) -> max -> sub -> exp -> sum -> div

This tests:
1. Canonicalization correctness
2. Reduction ordering (max before exp, sum after exp)
3. Numerical stability (subtract max prevents overflow)

Workload: softmax(x, axis=-1)
"""

import time
import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse
from kernel.lower import lower_region_to_kernel
from kernel.execute_ref import execute_kernel_ref
from kernel.candidate import generate_candidates, CandidateStrategy


# ------------------------------------------------------------
# Workloads
# ------------------------------------------------------------

def eager_softmax(x):
    """MLX's built-in softmax."""
    return mx.softmax(x, axis=-1)


def manual_softmax(x):
    """Numerically stable manual softmax."""
    max_x = mx.max(x, axis=-1, keepdims=True)
    exp_x = mx.exp(x - max_x)
    sum_exp = mx.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp


def unstable_softmax(x):
    """Naive softmax (numerically unstable for large values)."""
    exp_x = mx.exp(x)
    return exp_x / mx.sum(exp_x, axis=-1, keepdims=True)


# ------------------------------------------------------------
# Benchmark Configuration
# ------------------------------------------------------------

# Typical attention softmax shapes: (batch, heads, seq_len, seq_len)
SIZES = [
    (8, 8, 128, 128),    # Small attention
    (4, 12, 256, 256),   # Medium
    (2, 16, 512, 512),   # Large
    (1, 32, 1024, 1024), # XL attention
]
WARMUP_ITERS = 10
BENCH_ITERS = 50


# ------------------------------------------------------------
# Benchmark Helpers
# ------------------------------------------------------------

def time_it(fn, iters=BENCH_ITERS):
    """Time a function with proper MLX synchronization."""
    for _ in range(5):
        fn()
    mx.eval()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    mx.eval()
    end = time.perf_counter()
    return (end - start) / iters, out


def analyze_canonicalization(g):
    """Analyze how softmax was canonicalized."""
    op_counts = {}
    for op in g.ops:
        op_name = op.op
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

    print("\n  Canonicalized ops:")
    expected_ops = ['max', 'subtract', 'exp', 'sum', 'divide']
    for op_name in expected_ops:
        count = op_counts.get(op_name, 0)
        status = "FOUND" if count > 0 else "MISSING"
        print(f"    {op_name:12s}: {count:2d} ({status})")

    # Check if softmax was eliminated
    if 'softmax' in op_counts:
        print("    WARNING: softmax not fully canonicalized!")
    else:
        print("    softmax: eliminated (GOOD)")

    return op_counts


def run_benchmark(shape):
    """Run benchmark for a given shape."""
    print(f"\n{'='*60}")
    print(f"Shape: {shape}")
    print(f"Softmax axis: -1 (last dim = {shape[-1]})")
    print('='*60)

    x = mx.random.normal(shape)
    mx.eval(x)

    # Compile built-in softmax to see canonicalization
    g = trace_and_compile(eager_softmax, [x])
    print(f"\n  Total ops after canonicalization: {len(g.ops)}")

    # Analyze canonicalization
    op_counts = analyze_canonicalization(g)

    # Fuse regions
    regions = greedy_fuse(g)
    print(f"\n  Fusion regions: {len(regions)}")

    if not regions:
        print("  No fusion regions!")
        return

    region = regions[0]
    kernel = lower_region_to_kernel(g, region)
    inputs = {kernel.inputs[0]: x}

    # Generate candidates
    candidates = generate_candidates(kernel)
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)

    # Warmup
    for _ in range(WARMUP_ITERS):
        eager_softmax(x)
        manual_softmax(x)
        execute_kernel_ref(kernel, inputs)
        if ref_candidate:
            ref_candidate.fn(x)
    mx.eval()

    # Benchmark
    t_builtin, out_builtin = time_it(lambda: eager_softmax(x))
    t_manual, out_manual = time_it(lambda: manual_softmax(x))

    def run_kernel_ref():
        env = execute_kernel_ref(kernel, inputs)
        return env[kernel.outputs[0]]

    t_kernel_ref, out_kernel_ref = time_it(run_kernel_ref)

    t_generated = None
    out_generated = None
    if ref_candidate:
        t_generated, out_generated = time_it(lambda: ref_candidate.fn(x))

    # Correctness
    print("\n  Correctness:")
    print(f"    manual vs builtin : {mx.allclose(out_manual, out_builtin)}")
    if out_generated is not None:
        print(f"    generated vs builtin: {mx.allclose(out_generated, out_builtin, atol=1e-5)}")

    # Timing
    print("\n  Timing:")
    print(f"    MLX softmax (builtin): {t_builtin*1e3:8.3f} ms")
    print(f"    Manual softmax       : {t_manual*1e3:8.3f} ms")
    print(f"    KernelIR Ref         : {t_kernel_ref*1e3:8.3f} ms")
    if t_generated is not None:
        print(f"    Generated            : {t_generated*1e3:8.3f} ms")

    # Speedup
    print("\n  Speedup vs builtin:")
    print(f"    Manual       : {t_builtin/t_manual:.2f}x")
    print(f"    KernelIR Ref : {t_builtin/t_kernel_ref:.2f}x")
    if t_generated is not None:
        print(f"    Generated    : {t_builtin/t_generated:.2f}x")

    # Numerical stability check
    print("\n  Numerical stability test:")
    x_large = x * 100  # Scale up to test stability
    mx.eval(x_large)

    out_stable = eager_softmax(x_large)
    out_manual_large = manual_softmax(x_large)

    has_nan_stable = mx.any(mx.isnan(out_stable)).item()
    has_nan_manual = mx.any(mx.isnan(out_manual_large)).item()

    print(f"    Builtin softmax NaN: {has_nan_stable}")
    print(f"    Manual softmax NaN : {has_nan_manual}")

    if not has_nan_stable and not has_nan_manual:
        print("    Both implementations are numerically stable")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: Softmax Canonicalization")
    print("Tests: max -> sub -> exp -> sum -> div decomposition")
    print("="*60)

    for shape in SIZES:
        run_benchmark(shape)

    print("\n" + "="*60)
    print("Summary:")
    print("  - Softmax should be canonicalized to primitive ops")
    print("  - Max subtraction ensures numerical stability")
    print("  - Reduction barriers limit fusion opportunities")
    print("="*60)
