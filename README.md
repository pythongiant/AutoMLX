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

[ ] Add op classification map (ELEMENTWISE, REDUCTION, GEMM, BARRIER)
[ ] Implement greedy forward fusion with single-consumer and on-chip footprint checks
[ ] Implement `cost_model(region)` returning `(benefit, resource_penalty)`
[ ] Run local beam search over join/split choices
[ ] Pattern-match `matmul + add + softmax` → dispatch FlashAttention template
[ ] Lower other regions to Triton/Metal templates and run 5–10 candidate autotunes
[ ] Cache tuned schedules by `(op sequence, shapes, dtype, device)`