"""
Benchmark: Fused vs Eager Execution
===================================

Compares unfused (eager) execution against fused KernelIR execution.

This benchmark measures the overhead and potential benefits of the fusion system:
1. Tracing overhead
2. Fusion region construction
3. KernelIR dispatch overhead
4. Memory traffic reduction from fusion

Workloads designed to have clear fusion opportunities.
"""

import time
import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse
from fusion.regions import find_regions
from kernel.lower import lower_region_to_kernel
from kernel.execute_ref import execute_kernel_ref
from kernel.candidate import generate_candidates, CandidateStrategy
from runtime.execute_fused import run_fused_regions


# ------------------------------------------------------------
# Workloads with clear fusion opportunities
# ------------------------------------------------------------

def workload_chain(x):
    """Long elementwise chain - best case for fusion."""
    y = mx.abs(x)
    y = y + 1
    y = mx.sqrt(y)
    y = mx.exp(y)
    y = mx.log(y + 1)
    y = mx.tanh(y)
    return y


def workload_multi_use(x):
    """Multiple uses of intermediate - tests value reuse."""
    y = mx.exp(x)
    z = y + y  # y used twice
    w = y * z  # y used again
    return w


def workload_broadcast(x, y):
    """Broadcast operations - common in neural networks."""
    # x: (batch, features), y: (features,)
    z = x + y  # broadcast add
    z = z * y  # broadcast multiply
    return mx.sigmoid(z)


def workload_compound(x):
    """Compound expression that can be fused."""
    return mx.exp(mx.sin(x) + mx.cos(x))


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


def run_benchmark(name, eager_fn, inputs, input_names=None):
    """Run benchmark comparing eager vs fused execution."""
    print(f"\n{'='*60}")
    print(f"Workload: {name}")
    if input_names:
        for n, inp in zip(input_names, inputs):
            print(f"  {n}: shape={inp.shape}")
    else:
        for i, inp in enumerate(inputs):
            print(f"  input[{i}]: shape={inp.shape}")
    print('='*60)

    # Compile and analyze
    g = trace_and_compile(eager_fn, inputs)

    # Get regions before and after fusion
    regions_unfused = find_regions(g)
    regions_fused = greedy_fuse(g)

    print(f"\n  Graph analysis:")
    print(f"    Total ops: {len(g.ops)}")
    print(f"    Regions (unfused): {len(regions_unfused)}")
    print(f"    Regions (fused)  : {len(regions_fused)}")

    # Compute fusion ratio
    if len(regions_unfused) > 0:
        fusion_ratio = len(regions_fused) / len(regions_unfused)
        print(f"    Fusion ratio: {fusion_ratio:.2f} ({len(regions_unfused)} -> {len(regions_fused)})")

    # Lower fused regions to KernelIR
    kernels = []
    for region in regions_fused:
        try:
            kernel = lower_region_to_kernel(g, region)
            kernels.append(kernel)
        except Exception as e:
            print(f"    Failed to lower region: {e}")

    print(f"    KernelIR regions: {len(kernels)}")

    if not kernels:
        print("  No kernels generated!")
        return

    # Warmup
    for _ in range(WARMUP_ITERS):
        eager_fn(*inputs)
    mx.eval()

    # Benchmark eager
    t_eager, out_eager = time_it(lambda: eager_fn(*inputs))

    # Benchmark fused execution through runtime
    def run_fused():
        # Build input dict for all kernels
        env = {}
        for i, (inp, tid) in enumerate(zip(inputs, g.inputs)):
            env[tid] = inp
        # Execute all fused regions
        return run_fused_regions(g, regions_fused, env)

    t_fused, out_fused = time_it(run_fused)

    # Benchmark individual kernel execution
    kernel = kernels[0]
    kernel_inputs = {}
    for i, tid in enumerate(kernel.inputs):
        if tid in g.inputs:
            idx = g.inputs.index(tid)
            if idx < len(inputs):
                kernel_inputs[tid] = inputs[idx]
        elif i < len(inputs):
            kernel_inputs[tid] = inputs[i]

    # Fill any missing inputs
    for tid in kernel.inputs:
        if tid not in kernel_inputs:
            # Find the tensor in the graph
            if tid in g.tensors:
                t = g.tensors[tid]
                # Create a dummy input of the right shape
                kernel_inputs[tid] = mx.zeros(t.shape, dtype=t.dtype)

    def run_kernel_ref():
        env = execute_kernel_ref(kernel, kernel_inputs)
        return env[kernel.outputs[0]]

    # Only benchmark if we have all inputs
    t_kernel_ref = None
    if len(kernel_inputs) == len(kernel.inputs):
        for _ in range(WARMUP_ITERS):
            run_kernel_ref()
        mx.eval()
        t_kernel_ref, _ = time_it(run_kernel_ref)

    # Generate and benchmark candidates
    candidates = generate_candidates(kernel)
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)

    t_generated = None
    if ref_candidate and len(kernel.inputs) == 1:
        # Simple case: single input kernel
        for _ in range(WARMUP_ITERS):
            ref_candidate.fn(inputs[0])
        mx.eval()
        t_generated, _ = time_it(lambda: ref_candidate.fn(inputs[0]))

    # Results
    print(f"\n  Timing:")
    print(f"    Eager MLX           : {t_eager*1e3:8.3f} ms")
    print(f"    Fused regions       : {t_fused*1e3:8.3f} ms")
    if t_kernel_ref is not None:
        print(f"    KernelIR (single)   : {t_kernel_ref*1e3:8.3f} ms")
    if t_generated is not None:
        print(f"    Generated (single)  : {t_generated*1e3:8.3f} ms")

    # Overhead analysis
    print(f"\n  Overhead analysis:")
    overhead = (t_fused - t_eager) / t_eager * 100
    if overhead > 0:
        print(f"    Fused overhead: +{overhead:.1f}%")
    else:
        print(f"    Fused speedup: {-overhead:.1f}%")

    # Memory traffic estimation
    input_bytes = sum(inp.size * 4 for inp in inputs)
    output_bytes = out_eager.size * 4 if hasattr(out_eager, 'size') else 0

    # Unfused: each intermediate is materialized
    num_intermediates = len(g.ops) - 1  # Rough estimate
    unfused_traffic = input_bytes + output_bytes + num_intermediates * input_bytes

    # Fused: only input and output
    fused_traffic = input_bytes + output_bytes

    print(f"\n  Memory traffic (estimated):")
    print(f"    Unfused: {unfused_traffic / 1e6:.1f} MB")
    print(f"    Fused  : {fused_traffic / 1e6:.1f} MB")
    print(f"    Savings: {(unfused_traffic - fused_traffic) / unfused_traffic * 100:.1f}%")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: Fused vs Eager Execution")
    print("Measures fusion system overhead and benefits")
    print("="*60)

    for shape in SIZES:
        x = mx.random.normal(shape)
        mx.eval(x)

        # Test each workload
        run_benchmark("Long elementwise chain", workload_chain, [x])
        run_benchmark("Multi-use intermediate", workload_multi_use, [x])
        run_benchmark("Compound expression", workload_compound, [x])

    # Broadcast workload
    print("\n" + "="*60)
    print("Broadcast workloads")
    print("="*60)

    x = mx.random.normal((1024, 512))
    y = mx.random.normal((512,))
    mx.eval(x, y)
    run_benchmark("Broadcast ops", workload_broadcast, [x, y], ["x (batch, features)", "y (features)"])

    print("\n" + "="*60)
    print("Summary:")
    print("  - Fusion reduces memory traffic by avoiding intermediates")
    print("  - Overhead from tracing/dispatch should be minimal")
    print("  - Longer chains benefit more from fusion")
    print("="*60)
