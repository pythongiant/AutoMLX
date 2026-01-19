"""
Benchmark: KernelIR Reference vs Generated Kernels
==================================================

Compares the KernelIR reference executor against generated kernel candidates.

The reference executor is the "source of truth" - always correct but potentially slow.
Generated kernels (AI or rule-based) aim to be faster while maintaining correctness.

This benchmark measures:
1. Reference executor overhead
2. Generated kernel speedup
3. Different generation strategies
"""

import time
import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse
from kernel.lower import lower_region_to_kernel
from kernel.execute_ref import execute_kernel_ref
from kernel.candidate import generate_candidates, generate_single_candidate, CandidateStrategy


# ------------------------------------------------------------
# Workloads
# ------------------------------------------------------------

def workload_elementwise(x):
    """Simple elementwise - baseline."""
    return mx.exp(mx.sqrt(mx.abs(x) + 1))


def workload_complex(x):
    """More complex elementwise chain."""
    y = mx.tanh(x)
    z = mx.sigmoid(y)
    w = mx.exp(-z * z)
    return w + mx.log(mx.abs(y) + 1)


def workload_trig(x):
    """Trigonometric operations."""
    return mx.sin(x) * mx.cos(x) + mx.tan(x / 10)


def workload_power(x):
    """Power and exponential operations."""
    return mx.power(mx.abs(x) + 1, 0.5) * mx.exp(-x * x / 2)


# ------------------------------------------------------------
# Benchmark Configuration
# ------------------------------------------------------------

SIZES = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
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


def run_benchmark(name, eager_fn, shape):
    """Compare KernelIR reference vs generated kernels."""
    print(f"\n{'='*60}")
    print(f"Workload: {name} | Shape: {shape}")
    print('='*60)

    x = mx.random.normal(shape)
    mx.eval(x)

    # Compile
    g = trace_and_compile(eager_fn, [x])
    regions = greedy_fuse(g)

    if not regions:
        print("  No fusion regions!")
        return

    region = regions[0]
    kernel = lower_region_to_kernel(g, region)
    inputs = {kernel.inputs[0]: x}

    print(f"\n  Kernel info:")
    print(f"    Ops: {kernel.num_ops()}")
    print(f"    Inputs: {len(kernel.inputs)}")
    print(f"    Outputs: {len(kernel.outputs)}")

    # Generate all candidate strategies
    all_strategies = [
        CandidateStrategy.REFERENCE,
        CandidateStrategy.FUSED,
        CandidateStrategy.VECTORIZED,
        CandidateStrategy.UNROLLED,
    ]
    candidates = generate_candidates(kernel, strategies=all_strategies)

    print(f"    Candidates generated: {len(candidates)}")
    for c in candidates:
        print(f"      - {c.strategy.name}")

    # Warmup all paths
    for _ in range(WARMUP_ITERS):
        eager_fn(x)
        execute_kernel_ref(kernel, inputs)
        for c in candidates:
            c.fn(x)
    mx.eval()

    # Benchmark eager
    t_eager, out_eager = time_it(lambda: eager_fn(x))

    # Benchmark reference executor
    def run_kernel_ref():
        env = execute_kernel_ref(kernel, inputs)
        return env[kernel.outputs[0]]

    t_kernel_ref, out_kernel_ref = time_it(run_kernel_ref)

    # Benchmark each candidate
    candidate_results = {}
    for c in candidates:
        try:
            t, out = time_it(lambda c=c: c.fn(x))
            correct = mx.allclose(out, out_eager, atol=1e-5)
            candidate_results[c.strategy.name] = (t, out, correct)
        except Exception as e:
            candidate_results[c.strategy.name] = (None, None, False)
            print(f"    {c.strategy.name} failed: {e}")

    # Results
    print(f"\n  Correctness:")
    print(f"    KernelIR Ref vs Eager: {mx.allclose(out_kernel_ref, out_eager)}")
    for name, (t, out, correct) in candidate_results.items():
        if t is not None:
            print(f"    {name:12s} vs Eager: {correct}")

    print(f"\n  Timing (ms):")
    print(f"    {'Method':<20s} {'Time':>10s} {'vs Eager':>12s} {'vs KernelRef':>12s}")
    print(f"    {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    print(f"    {'Eager MLX':<20s} {t_eager*1e3:>10.3f} {'-':>12s} {'-':>12s}")
    print(f"    {'KernelIR Ref':<20s} {t_kernel_ref*1e3:>10.3f} {t_eager/t_kernel_ref:>11.2f}x {'-':>12s}")

    for name, (t, _, correct) in candidate_results.items():
        if t is not None:
            vs_eager = t_eager / t
            vs_ref = t_kernel_ref / t
            status = "" if correct else " (!)"
            print(f"    {name:<20s} {t*1e3:>10.3f} {vs_eager:>11.2f}x {vs_ref:>11.2f}x{status}")

    # Best candidate analysis
    valid_candidates = [(name, t) for name, (t, _, correct) in candidate_results.items() if t is not None and correct]
    if valid_candidates:
        best_name, best_t = min(valid_candidates, key=lambda x: x[1])
        print(f"\n  Best candidate: {best_name}")
        print(f"    Speedup vs KernelIR Ref: {t_kernel_ref/best_t:.2f}x")
        print(f"    Speedup vs Eager: {t_eager/best_t:.2f}x")

    # Show generated kernel source for reference
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)
    if ref_candidate:
        print(f"\n  Reference kernel source:")
        for i, line in enumerate(ref_candidate.source.split('\n')[:15]):
            print(f"    {line}")
        if len(ref_candidate.source.split('\n')) > 15:
            print(f"    ... ({len(ref_candidate.source.split(chr(10)))} lines total)")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: KernelIR Reference vs Generated Kernels")
    print("Comparing execution strategies")
    print("="*60)

    workloads = [
        ("Elementwise", workload_elementwise),
        ("Complex chain", workload_complex),
        ("Trigonometric", workload_trig),
        ("Power/Exp", workload_power),
    ]

    for shape in SIZES:
        print(f"\n\n{'#'*60}")
        print(f"# Size: {shape}")
        print('#'*60)

        for name, fn in workloads:
            run_benchmark(name, fn, shape)

    print("\n" + "="*60)
    print("Summary:")
    print("  - Reference executor: always correct, baseline performance")
    print("  - FUSED strategy: uses mx.compile for MLX-level fusion")
    print("  - Generated speedup depends on workload complexity")
    print("="*60)
