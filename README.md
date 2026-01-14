# Why do we need Auto-Perfect
MLX is possibly the go-to for local MLX inference given how much apple has been investing in it. Similar tools exist for Nvidia CUDA to write custom CUDA kernels.

# How does this work?
```
MLX Code
  ↓
Graph Capture
  ↓
Fusion Detection
  ↓
Kernel IR
  ↓
LLM Kernel Generation
  ↓
Compile + Repair
  ↓
Validate
  ↓
Benchmark
  ↓
Cache
  ↓
Runtime Dispatch
```

## Wait hol'up why do we need custom kernels?
We need custom GPU kernels because general-purpose kernels leave a lot of performance, memory efficiency, and control on the table. Once you push beyond “standard deep learning workloads,” the defaults become a bottleneck.

### Kernel Fusion
Custom GPU kernels are how you turn mathematical insight into hardware-level speed. frameworks give correctness, kernels give dominance.

Every GPU kernel launch has overhead and memory traffic.

Typical PyTorch code:

```
load A → kernel
load B → kernel
add → kernel
relu → kernel
store
```

That’s:
- Multiple kernel launches
- Multiple global memory reads/writes

A custom kernel can do:
```
load once → compute everything → store once
```
#### Why this matters

GPU performance is usually memory-bound, not FLOP-bound.

Fusing ops:
- Reduces global memory traffic
- Keeps data in registers / shared memory
- Improves cache locality

FlashAttention exists almost entirely because of this.
# Checklist

[x] Identity-safe tensor tracing using `id(tensor)` (no weakrefs, view-safe)  
[x] Full op surface tracing across elementwise, reductions, linalg, shape, indexing, and multi-output ops  
[x] Correct SSA-style GraphIR with explicit producers, consumers, constants, and kwargs  
[x] Faithful runtime interpreter supporting unary, binary, variadic, kwargs, and multi-output ops  
[x] Deterministic input binding captured at trace time (no ordering or shape inference)  
[x] Validation suite covering elementwise, reductions, GEMM, broadcasting, views, and multi-output semantics  
[x] End-to-end correctness parity vs eager MLX (`mx.allclose`)  

[x] Add op classification map (ELEMENTWISE, REDUCTION, GEMM, RESHAPE/VIEW, INDEXING, BARRIER)  
[x] Define conservative fusion barriers (IO, random, sort/select, multi-output shape ops)  

[x] Canonicalize softmax (`max → sub → exp → sum → div`) with numerically stable lowering  
[x] Eliminate `softmax` op from IR and expose attention-friendly patterns  
[x] Add structural + numerical regression tests for softmax canonicalization  

[ ] Implement greedy forward fusion with single-consumer constraint and alias-safety checks  
[ ] Add legality checks (reduction boundaries, barriers, multi-output splits)  
[ ] Track on-chip footprint (register/shared memory) per fusion region  
[ ] Implement `cost_model(region) -> (benefit, resource_penalty)`  
[ ] Run local beam search over region join/split decisions  

[ ] Pattern-match `matmul + add + max + sub + exp + sum + div` → dispatch FlashAttention-style template  
[ ] Lower remaining regions to Triton / Metal templates  
[ ] Autotune 5–10 candidate schedules per region (tile sizes, unroll, vector width)  
[ ] Cache tuned schedules by `(op sequence, shapes, dtype, device)`  

[ ] Add regression tests for fusion legality and numerical stability  
[ ] Add performance benchmarks vs unfused MLX execution
