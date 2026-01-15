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

# Core Compiler (Deterministic)

[x] Identity-safe tensor tracing using id(tensor)
[x] Full op surface tracing (elementwise, reduction, linalg, shape, multi-output)
[x] SSA-style GraphIR with explicit producers, consumers, constants, kwargs
[x] Deterministic input binding at trace time
[x] Faithful runtime interpreter (unfused execution)
[x] End-to-end correctness vs eager MLX

# Canonicalization & Semantics

[x] Op classification map (ELEMENTWISE, REDUCTION, GEMM, RESHAPE, INDEXING, BARRIER)
[x] Conservative fusion barriers defined
[x] Numerically stable softmax canonicalization
[x] Softmax eliminated from IR
[x] Structural + numerical tests for canonicalization

# Fusion (Compiler-Owned)

[x] Legality-only fusion region discovery
[x] SSA single-consumer enforcement
[x] Reduction boundary enforcement
[x] Barrier enforcement
[x] Deterministic toposort-based region construction

[x] Greedy forward fusion
[x] Incremental cost gating (Δbenefit > Δpenalty)
[x] Peak live-byte tracking
[x] Hard footprint caps
[x] Cost model validated against pathological cases

# Kernel Interface Layer (New, Critical)

[x] Explicit FusedRegion abstraction
[x] Stable kernel input/output contract per region
[x] Region signature hashing (ops, shapes, dtypes)

[ ] Define KernelIR (lower than GraphIR, higher than MLX code)
[ ] Define kernel ABI (inputs, outputs, temporaries)

# AI-Generated Kernel Layer

[ ] Serialize FusedRegion → prompt representation
[ ] Constrain generation to:

pure MLX

no side effects

no graph mutation

[ ] Generate candidate kernels for a fused region
[ ] Enforce shape/dtype guards in generated code
[ ] Static validation of generated kernel (signature, outputs)
[ ] Run correctness check vs reference interpreter

[ ] (Optional) Generate multiple variants per region
[ ] Score variants via:

cost model estimate

microbenchmark timing

[ ] Select best kernel deterministically

Caching & Reuse

[ ] Cache generated kernels by:

(region_ops, shapes, dtypes, device)


[ ] Reuse kernels across graphs and runs
[ ] Invalidate cache on compiler version change

# Execution & Integration

[x] Fallback execution for fused regions
[ ] Dispatch AI-generated kernel if available
[ ] Fallback to interpreter if generation fails
[ ] Preserve debuggability (region → kernel mapping)

# Validation & Benchmarking

[x] Fusion legality regression tests
[x] Cost-model regression tests

[ ] Generated kernel correctness tests
[ ] Numerical stability tests
[ ] Performance benchmarks:

unfused vs fused

fused vs AI-generated

cold vs cached kernels

---