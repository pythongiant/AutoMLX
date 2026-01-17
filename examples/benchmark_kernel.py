"""
Benchmark:
1) Eager MLX
2) KernelIR reference executor (fused fallback)
3) Hand-written generated kernel

Goal:
- Measure real execution time
- Verify correctness
- No dispatch abstraction hiding costs
"""

import time
import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse
from kernel.lower import lower_region_to_kernel
from kernel.execute_ref import execute_kernel_ref


# ------------------------------------------------------------
# Workload
# ------------------------------------------------------------

def eager_fn(x):
    return mx.exp(mx.sqrt(mx.abs(x) + 1))


def generated_kernel(t0):
    # what an LLM would generate
    t1 = mx.abs(t0)
    t2 = mx.sqrt(t1 + 1)
    t3 = mx.exp(t2)
    return t3


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

x = mx.random.normal((2048, 2048))
mx.eval(x)  # materialize

# Compile
g = trace_and_compile(eager_fn, [x])
regions = greedy_fuse(g)
region = regions[0]
kernel = lower_region_to_kernel(g, region)

inputs = {kernel.inputs[0]: x}


# ------------------------------------------------------------
# Warmup (important for MLX)
# ------------------------------------------------------------

for _ in range(5):
    eager_fn(x)
    generated_kernel(x)
    execute_kernel_ref(kernel, inputs)

mx.eval()


# ------------------------------------------------------------
# Benchmark helpers
# ------------------------------------------------------------

def time_it(fn, iters=20):
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    mx.eval()  # force sync
    end = time.perf_counter()
    return (end - start) / iters, out


# ------------------------------------------------------------
# 1) Eager MLX
# ------------------------------------------------------------

t_eager, out_eager = time_it(lambda: eager_fn(x))


# ------------------------------------------------------------
# 2) KernelIR reference executor
# ------------------------------------------------------------

def run_kernel_ref():
    env = execute_kernel_ref(kernel, inputs)
    return env[kernel.outputs[0]]

t_kernel_ref, out_kernel_ref = time_it(run_kernel_ref)


# ------------------------------------------------------------
# 3) Generated kernel
# ------------------------------------------------------------

t_generated, out_generated = time_it(lambda: generated_kernel(x))


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n=== Correctness ===")
print("kernel_ref == eager :", mx.allclose(out_kernel_ref, out_eager))
print("generated == eager  :", mx.allclose(out_generated, out_eager))

print("\n=== Timing (avg per run) ===")
print(f"Eager MLX          : {t_eager * 1e3:.3f} ms")
print(f"KernelIR fallback  : {t_kernel_ref * 1e3:.3f} ms")
print(f"Generated kernel   : {t_generated * 1e3:.3f} ms")

print("\n=== Speedups ===")
print(f"Generated / Eager        : {t_eager / t_generated:.2f}×")
print(f"Generated / KernelRef    : {t_kernel_ref / t_generated:.2f}×")
