"""
Benchmark: Cache Effects (Cold vs Warm)
=======================================

Measures the impact of caching on generated kernel performance.

Cold execution includes:
- Kernel compilation/generation
- Cache miss overhead
- First-run JIT compilation

Warm execution benefits from:
- Cached kernels
- JIT-compiled code paths
- Memory allocation patterns

This is critical for understanding real-world performance.
"""

import time
import gc
import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse
from kernel.lower import lower_region_to_kernel
from kernel.execute_ref import execute_kernel_ref
from kernel.candidate import generate_candidates, CandidateStrategy
from kernel.generated.cache import put_kernel, clear_cache, get_kernel, kernel_signature
from kernel.generated.dispatch import execute_kernel


# ------------------------------------------------------------
# Workloads
# ------------------------------------------------------------

def workload_simple(x):
    """Simple workload for cache testing."""
    return mx.exp(mx.sqrt(mx.abs(x) + 1))


def workload_medium(x):
    """Medium complexity workload."""
    y = mx.tanh(x)
    z = mx.sigmoid(y)
    return mx.exp(-z * z) + mx.log(mx.abs(y) + 1)


def workload_complex(x):
    """Complex workload with many ops."""
    y = mx.sin(x) * mx.cos(x)
    z = mx.tanh(y) + mx.sigmoid(y)
    w = mx.exp(-z * z / 2)
    return w * mx.sqrt(mx.abs(z) + 1)


# ------------------------------------------------------------
# Benchmark Configuration
# ------------------------------------------------------------

SHAPE = (2048, 2048)
COLD_RUNS = 5  # Number of cold runs to average
WARM_ITERS = 50


# ------------------------------------------------------------
# Benchmark Helpers
# ------------------------------------------------------------

def time_single(fn):
    """Time a single execution with sync."""
    start = time.perf_counter()
    out = fn()
    mx.eval()
    end = time.perf_counter()
    return end - start, out


def time_batch(fn, iters):
    """Time multiple executions."""
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    mx.eval()
    end = time.perf_counter()
    return (end - start) / iters, out


def clear_all_caches():
    """Clear all relevant caches for cold benchmarking."""
    # Clear kernel cache
    try:
        clear_cache()
    except:
        pass

    # Force garbage collection
    gc.collect()

    # Clear MLX caches by creating memory pressure
    # (MLX may cache compiled graphs)
    dummy = mx.zeros((1000, 1000))
    mx.eval(dummy)
    del dummy
    gc.collect()


def run_benchmark(name, eager_fn):
    """Run cold vs warm benchmark for a workload."""
    print(f"\n{'='*60}")
    print(f"Workload: {name}")
    print(f"Shape: {SHAPE}")
    print('='*60)

    x = mx.random.normal(SHAPE)
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
    sig = kernel_signature(kernel)

    print(f"\n  Kernel: {kernel.num_ops()} ops")
    print(f"  Signature: {sig[:50]}...")

    # Generate candidate
    candidates = generate_candidates(kernel)
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)

    if not ref_candidate:
        print("  No reference candidate!")
        return

    # -------------------------
    # Cold benchmarks
    # -------------------------
    print(f"\n  Cold execution ({COLD_RUNS} runs):")

    # Eager cold
    clear_all_caches()
    cold_eager_times = []
    for i in range(COLD_RUNS):
        t, _ = time_single(lambda: eager_fn(x))
        cold_eager_times.append(t)
        # Don't clear between runs to see warmup effect
    print(f"    Eager cold runs: {[f'{t*1e3:.2f}ms' for t in cold_eager_times]}")

    # KernelIR ref cold
    clear_all_caches()
    cold_ref_times = []
    for i in range(COLD_RUNS):
        def run_ref():
            env = execute_kernel_ref(kernel, inputs)
            return env[kernel.outputs[0]]
        t, _ = time_single(run_ref)
        cold_ref_times.append(t)
    print(f"    KernelIR cold runs: {[f'{t*1e3:.2f}ms' for t in cold_ref_times]}")

    # Generated cold
    clear_all_caches()
    cold_gen_times = []
    for i in range(COLD_RUNS):
        t, _ = time_single(lambda: ref_candidate.fn(x))
        cold_gen_times.append(t)
    print(f"    Generated cold runs: {[f'{t*1e3:.2f}ms' for t in cold_gen_times]}")

    # Dispatch with cache miss
    clear_all_caches()
    cold_dispatch_times = []
    for i in range(COLD_RUNS):
        t, _ = time_single(lambda: execute_kernel(kernel, inputs))
        cold_dispatch_times.append(t)
    print(f"    Dispatch cold runs: {[f'{t*1e3:.2f}ms' for t in cold_dispatch_times]}")

    # -------------------------
    # Warm benchmarks
    # -------------------------
    print(f"\n  Warm execution ({WARM_ITERS} iterations):")

    # Register kernel for warm dispatch
    put_kernel(sig, ref_candidate.fn)

    # Warmup all paths
    for _ in range(10):
        eager_fn(x)
        execute_kernel_ref(kernel, inputs)
        ref_candidate.fn(x)
        execute_kernel(kernel, inputs)
    mx.eval()

    # Benchmark warm
    t_eager_warm, out_eager = time_batch(lambda: eager_fn(x), WARM_ITERS)

    def run_ref():
        env = execute_kernel_ref(kernel, inputs)
        return env[kernel.outputs[0]]
    t_ref_warm, out_ref = time_batch(run_ref, WARM_ITERS)

    t_gen_warm, out_gen = time_batch(lambda: ref_candidate.fn(x), WARM_ITERS)
    t_dispatch_warm, out_dispatch = time_batch(lambda: execute_kernel(kernel, inputs), WARM_ITERS)

    print(f"    Eager warm       : {t_eager_warm*1e3:8.3f} ms/iter")
    print(f"    KernelIR warm    : {t_ref_warm*1e3:8.3f} ms/iter")
    print(f"    Generated warm   : {t_gen_warm*1e3:8.3f} ms/iter")
    print(f"    Dispatch warm    : {t_dispatch_warm*1e3:8.3f} ms/iter")

    # -------------------------
    # Analysis
    # -------------------------
    print(f"\n  Cold vs Warm analysis:")

    # First cold vs warm
    first_cold_eager = cold_eager_times[0]
    cold_warm_ratio_eager = first_cold_eager / t_eager_warm
    print(f"    Eager: first cold {first_cold_eager*1e3:.2f}ms, warm {t_eager_warm*1e3:.2f}ms ({cold_warm_ratio_eager:.1f}x slower cold)")

    first_cold_gen = cold_gen_times[0]
    cold_warm_ratio_gen = first_cold_gen / t_gen_warm
    print(f"    Generated: first cold {first_cold_gen*1e3:.2f}ms, warm {t_gen_warm*1e3:.2f}ms ({cold_warm_ratio_gen:.1f}x slower cold)")

    # Warmup progression
    print(f"\n  Warmup progression (generated):")
    for i, t in enumerate(cold_gen_times):
        speedup = cold_gen_times[0] / t
        print(f"    Run {i+1}: {t*1e3:.2f}ms ({speedup:.2f}x vs first)")

    # Dispatch overhead
    dispatch_overhead = (t_dispatch_warm - t_gen_warm) / t_gen_warm * 100
    print(f"\n  Dispatch overhead (warm): {dispatch_overhead:+.1f}%")

    # Correctness
    print(f"\n  Correctness:")
    print(f"    Generated vs Eager: {mx.allclose(out_gen, out_eager)}")
    print(f"    Dispatch vs Eager: {mx.allclose(out_dispatch[kernel.outputs[0]], out_eager)}")


# ------------------------------------------------------------
# Cache hit rate benchmark
# ------------------------------------------------------------

def benchmark_cache_hits():
    """Benchmark cache hit rate impact."""
    print(f"\n{'='*60}")
    print("Cache Hit Rate Impact")
    print('='*60)

    x = mx.random.normal(SHAPE)
    mx.eval(x)

    # Create multiple kernels
    workloads = [
        ("w1", workload_simple),
        ("w2", workload_medium),
        ("w3", workload_complex),
    ]

    kernels_and_candidates = []
    for name, fn in workloads:
        g = trace_and_compile(fn, [x])
        regions = greedy_fuse(g)
        if regions:
            kernel = lower_region_to_kernel(g, regions[0])
            candidates = generate_candidates(kernel)
            ref = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)
            if ref:
                kernels_and_candidates.append((name, kernel, ref))

    # Clear and register all
    clear_all_caches()
    for name, kernel, candidate in kernels_and_candidates:
        sig = kernel_signature(kernel)
        put_kernel(sig, candidate.fn)

    print(f"\n  Registered {len(kernels_and_candidates)} kernels")

    # Warmup
    for _ in range(10):
        for name, kernel, candidate in kernels_and_candidates:
            candidate.fn(x)
    mx.eval()

    # Interleaved execution (tests cache switching)
    print(f"\n  Interleaved execution (random order):")
    import random

    iterations = 100
    order = []
    for _ in range(iterations):
        order.extend(range(len(kernels_and_candidates)))
    random.shuffle(order)

    start = time.perf_counter()
    for i in order:
        name, kernel, candidate = kernels_and_candidates[i]
        candidate.fn(x)
    mx.eval()
    end = time.perf_counter()

    interleaved_time = (end - start) / len(order)
    print(f"    Avg time per kernel: {interleaved_time*1e3:.3f} ms")

    # Sequential execution (same kernel repeated)
    print(f"\n  Sequential execution (same kernel repeated):")
    for name, kernel, candidate in kernels_and_candidates:
        start = time.perf_counter()
        for _ in range(iterations):
            candidate.fn(x)
        mx.eval()
        end = time.perf_counter()
        seq_time = (end - start) / iterations
        print(f"    {name}: {seq_time*1e3:.3f} ms/iter")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: Cache Effects (Cold vs Warm)")
    print("="*60)

    # Individual workload benchmarks
    run_benchmark("Simple", workload_simple)
    run_benchmark("Medium", workload_medium)
    run_benchmark("Complex", workload_complex)

    # Cache hit rate benchmark
    benchmark_cache_hits()

    print("\n" + "="*60)
    print("Summary:")
    print("  - Cold execution can be 2-10x slower than warm")
    print("  - MLX JIT compilation adds significant first-run overhead")
    print("  - Kernel caching is essential for performance")
    print("  - Dispatch overhead should be <5% for warm kernels")
    print("="*60)
