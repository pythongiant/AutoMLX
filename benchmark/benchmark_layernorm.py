"""
Benchmark: LayerNorm Operations
===============================

Fusion stress test with reduction + broadcast patterns.

LayerNorm is a key pattern that combines:
- Reduction (mean, variance)
- Broadcast (subtract mean, divide by std)
- Elementwise (add epsilon, sqrt)

This tests whether fusion can eliminate intermediate materialization.

Workload: (x - mean(x)) / sqrt(var(x) + eps)
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

def eager_layernorm(x, eps=1e-5):
    """Manual layernorm implementation."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def eager_layernorm_affine(x, weight, bias, eps=1e-5):
    """LayerNorm with learnable parameters."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / mx.sqrt(var + eps)
    return normed * weight + bias


def eager_rmsnorm(x, eps=1e-5):
    """RMSNorm - simpler variant without mean centering."""
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


# ------------------------------------------------------------
# Benchmark Configuration
# ------------------------------------------------------------

# Typical transformer shapes: (batch, seq_len, hidden_dim)
SIZES = [
    (32, 128, 768),     # Small BERT-like
    (16, 512, 1024),    # Medium
    (8, 1024, 2048),    # Large
    (4, 2048, 4096),    # XL
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


def run_benchmark(name, eager_fn, shape, extra_inputs=None):
    """Run benchmark for a single workload."""
    print(f"\n{'-'*50}")
    print(f"Workload: {name} | Shape: {shape}")
    print('-'*50)

    x = mx.random.normal(shape)
    mx.eval(x)

    if extra_inputs is None:
        inputs_list = [x]
        all_inputs = (x,)
    else:
        inputs_list = [x] + extra_inputs
        all_inputs = (x,) + tuple(extra_inputs)

    # Compile
    g = trace_and_compile(eager_fn, inputs_list)
    regions = greedy_fuse(g)

    print(f"  Fusion regions: {len(regions)}")
    print(f"  Total ops in graph: {len(g.ops)}")

    if not regions:
        print("  No fusion regions found!")
        return

    # We may have multiple regions - benchmark each
    region = regions[0]
    kernel = lower_region_to_kernel(g, region)

    print(f"  First region ops: {kernel.num_ops()}")

    # Build kernel inputs
    kernel_inputs = {}
    for i, tid in enumerate(kernel.inputs):
        if i < len(inputs_list):
            kernel_inputs[tid] = inputs_list[i]

    # Generate candidates
    candidates = generate_candidates(kernel)
    ref_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.REFERENCE), None)
    fused_candidate = next((c for c in candidates if c.strategy == CandidateStrategy.FUSED), None)

    # Warmup
    for _ in range(WARMUP_ITERS):
        eager_fn(*all_inputs)
        if kernel_inputs:
            execute_kernel_ref(kernel, kernel_inputs)
    mx.eval()

    # Benchmark eager
    t_eager, out_eager = time_it(lambda: eager_fn(*all_inputs))

    # Benchmark with MLX's built-in layernorm for comparison
    t_builtin = None
    if name == "LayerNorm":
        def mlx_layernorm():
            # MLX doesn't have built-in layernorm, use fast_layer_norm if available
            try:
                return mx.fast.layer_norm(x, weight=None, bias=None, eps=1e-5)
            except:
                return eager_fn(*all_inputs)

        try:
            t_builtin, _ = time_it(mlx_layernorm)
        except:
            pass

    # Benchmark kernel ref (only if we have proper inputs)
    t_kernel_ref = None
    if kernel_inputs and len(kernel_inputs) == len(kernel.inputs):
        def run_kernel_ref():
            env = execute_kernel_ref(kernel, kernel_inputs)
            return env[kernel.outputs[0]]

        try:
            t_kernel_ref, _ = time_it(run_kernel_ref)
        except Exception as e:
            print(f"  KernelIR execution failed: {e}")

    # Results
    print("\n  Timing:")
    print(f"    Eager (manual)  : {t_eager*1e3:8.3f} ms")
    if t_builtin is not None:
        print(f"    MLX built-in    : {t_builtin*1e3:8.3f} ms")
    if t_kernel_ref is not None:
        print(f"    KernelIR Ref    : {t_kernel_ref*1e3:8.3f} ms")

    # Memory traffic analysis
    input_bytes = x.size * 4  # float32
    output_bytes = out_eager.size * 4

    # LayerNorm reads x twice (mean, then normalized), plus intermediates
    estimated_reads = input_bytes * 2  # Conservative
    total_bytes = estimated_reads + output_bytes

    bandwidth = total_bytes / t_eager / 1e9
    print(f"\n  Estimated bandwidth: {bandwidth:.1f} GB/s")

    # Fusion benefit analysis
    print("\n  Fusion analysis:")
    print(f"    Ops that could be fused: {len(g.ops)}")
    print(f"    Actual fused regions: {len(regions)}")
    if len(regions) > 1:
        print("    Note: Reductions create fusion barriers")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Benchmark: LayerNorm Operations")
    print("Fusion stress test: reduction + broadcast patterns")
    print("="*60)

    for shape in SIZES:
        # Basic LayerNorm
        run_benchmark("LayerNorm", eager_layernorm, shape)

        # RMSNorm (simpler, often used in modern models)
        run_benchmark("RMSNorm", eager_rmsnorm, shape)

    print("\n" + "="*60)
    print("Summary: LayerNorm combines reduction + broadcast.")
    print("Fusion can eliminate intermediate materialization.")
    print("="*60)
