"""
Benchmark: Elementwise Operations
=================================

Sanity check benchmark - expect NO speedup from fusion.

Elementwise ops are already efficient in MLX; the goal is to verify:
1. KernelIR reference executor has acceptable overhead
2. Generated kernel matches eager execution
3. No significant regression from compilation

Workload: exp(sqrt(abs(x) + 1))
"""

import time
import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse
from kernel.lower import lower_region_to_kernel
from kernel.execute_ref import execute_kernel_ref
from kernel.candidate import generate_candidates, CandidateStrategy


# ------------------------------------------------------------
# Workload
# ------------------------------------------------------------

def eager_fn(x):
    """Simple elementwise chain - already optimized by MLX."""
    return mx.exp(mx.sqrt(mx.abs(x) + 1))


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
    # Warmup
    for _ in range(5):
        fn()
    mx.eval()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    mx.eval()  # Force GPU sync
    end = time.perf_counter()
    return (end - start) / iters, out


def run_benchmark(shape):
    """Run benchmark for a given input shape."""
    print(f"\n{'='*60}")
    print(f"Shape: {shape}")
    print('='*60)

    # Setup input
    x = mx.random.normal(shape)
    mx.eval(x)

    # Compile
    g = trace_and_compile(eager_fn, [x])
    regions = greedy_fuse(g)

    if not regions:
        print("No fusion regions found!")
        return

    region = regions[0]
    kernel = lower_region_to_kernel(g, region)
    inputs = {kernel.inputs[0]: x}

    # Generate candidates
    candidates = generate_candidates(kernel)
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)
    fused_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.FUSED), None)

    # Warmup all paths
    for _ in range(WARMUP_ITERS):
        eager_fn(x)
        execute_kernel_ref(kernel, inputs)
        if ref_candidate:
            ref_candidate.fn(x)
        if fused_candidate:
            fused_candidate.fn(x)
    mx.eval()

    # Benchmark
    t_eager, out_eager = time_it(lambda: eager_fn(x))

    def run_kernel_ref():
        env = execute_kernel_ref(kernel, inputs)
        return env[kernel.outputs[0]]

    t_kernel_ref, out_kernel_ref = time_it(run_kernel_ref)

    results = {
        "Eager MLX": (t_eager, out_eager),
        "KernelIR Ref": (t_kernel_ref, out_kernel_ref),
    }

    if ref_candidate:
        t_ref_cand, out_ref_cand = time_it(lambda: ref_candidate.fn(x))
        results["Generated (ref)"] = (t_ref_cand, out_ref_cand)

    if fused_candidate:
        t_fused_cand, out_fused_cand = time_it(lambda: fused_candidate.fn(x))
        results["Generated (fused)"] = (t_fused_cand, out_fused_cand)

    # Correctness checks
    print("\n--- Correctness ---")
    for name, (_, out) in results.items():
        if name == "Eager MLX":
            continue
        match = mx.allclose(out, out_eager)
        status = "PASS" if match else "FAIL"
        print(f"  {name}: {status}")

    # Timing results
    print("\n--- Timing (avg per call) ---")
    for name, (t, _) in results.items():
        print(f"  {name:20s}: {t*1e3:8.3f} ms")

    # Speedups (relative to eager)
    print("\n--- Speedup vs Eager ---")
    for name, (t, _) in results.items():
        if name == "Eager MLX":
            continue
        speedup = t_eager / t
        indicator = "faster" if speedup > 1.0 else "slower"
        print(f"  {name:20s}: {speedup:6.2f}x ({indicator})")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: Elementwise Operations")
    print("Workload: exp(sqrt(abs(x) + 1))")
    print("Expected: No significant speedup (MLX already optimizes)")
    print("="*60)

    for shape in SIZES:
        run_benchmark(shape)

    print("\n" + "="*60)
    print("Summary: Elementwise ops are memory-bound.")
    print("Fusion overhead should be minimal but not beneficial.")
    print("="*60)
