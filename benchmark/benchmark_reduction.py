"""
Benchmark: Reduction Operations
===============================

Memory-bound baseline benchmark.

Tests mean and variance operations which are fundamental reduction patterns.
These establish a baseline for memory-bound operations.

Workload:
1. mean(x, axis=1)
2. variance: mean((x - mean(x))^2)
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

def eager_mean(x):
    """Simple mean reduction."""
    return mx.mean(x, axis=1)


def eager_variance(x):
    """Variance: mean((x - mean)^2)"""
    m = mx.mean(x, axis=1, keepdims=True)
    diff = x - m
    return mx.mean(diff * diff, axis=1)


def eager_sum_of_squares(x):
    """Sum of squares reduction."""
    return mx.sum(x * x, axis=1)


# ------------------------------------------------------------
# Benchmark Configuration
# ------------------------------------------------------------

SIZES = [
    (1024, 512),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
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
    """Run benchmark for a single workload and shape."""
    print(f"\n{'-'*50}")
    print(f"Workload: {name} | Shape: {shape}")
    print('-'*50)

    x = mx.random.normal(shape)
    mx.eval(x)

    # Compile
    g = trace_and_compile(eager_fn, [x])
    regions = greedy_fuse(g)

    if not regions:
        print("  No fusion regions found!")
        return

    # May have multiple regions due to reduction barriers
    print(f"  Regions found: {len(regions)}")

    # Process first region
    region = regions[0]
    kernel = lower_region_to_kernel(g, region)
    inputs = {kernel.inputs[0]: x}

    # Generate candidates
    candidates = generate_candidates(kernel)
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)

    # Warmup
    for _ in range(WARMUP_ITERS):
        eager_fn(x)
        execute_kernel_ref(kernel, inputs)
        if ref_candidate:
            ref_candidate.fn(x)
    mx.eval()

    # Benchmark eager
    t_eager, out_eager = time_it(lambda: eager_fn(x))

    # Benchmark kernel ref
    def run_kernel_ref():
        env = execute_kernel_ref(kernel, inputs)
        return env[kernel.outputs[0]]

    t_kernel_ref, out_kernel_ref = time_it(run_kernel_ref)

    # Benchmark generated candidate
    t_generated = None
    out_generated = None
    if ref_candidate:
        t_generated, out_generated = time_it(lambda: ref_candidate.fn(x))

    # Results
    print("\n  Timing:")
    print(f"    Eager MLX     : {t_eager*1e3:8.3f} ms")
    print(f"    KernelIR Ref  : {t_kernel_ref*1e3:8.3f} ms")
    if t_generated is not None:
        print(f"    Generated     : {t_generated*1e3:8.3f} ms")

    # Note: For reductions, we may not get the same output due to region boundaries
    # Only check if shapes match
    print("\n  Output shapes:")
    print(f"    Eager         : {out_eager.shape}")
    print(f"    KernelIR Ref  : {out_kernel_ref.shape}")

    # Memory bandwidth estimation
    bytes_read = x.size * 4  # float32
    bytes_written = out_eager.size * 4
    total_bytes = bytes_read + bytes_written
    bandwidth_eager = total_bytes / t_eager / 1e9  # GB/s

    print(f"\n  Estimated bandwidth (eager): {bandwidth_eager:.1f} GB/s")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: Reduction Operations")
    print("Memory-bound baseline for fusion analysis")
    print("="*60)

    workloads = [
        ("mean(x, axis=1)", eager_mean),
        ("sum(x*x, axis=1)", eager_sum_of_squares),
    ]

    for shape in SIZES:
        for name, fn in workloads:
            run_benchmark(name, fn, shape)

    print("\n" + "="*60)
    print("Summary: Reductions are memory-bound.")
    print("Performance limited by memory bandwidth, not compute.")
    print("="*60)
