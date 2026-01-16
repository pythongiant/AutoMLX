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

* [x] Identity-safe tensor tracing using `id(tensor)`
* [x] Full op surface tracing (elementwise, reductions, linalg, shape, indexing, multi-output)
* [x] SSA-style GraphIR with explicit producers, consumers, constants, and kwargs
* [x] Deterministic input binding captured at trace time
* [x] Faithful runtime interpreter for unfused GraphIR execution
* [x] End-to-end numerical correctness vs eager MLX (`mx.allclose`)

---

# Canonicalization & Semantics

* [x] Op classification map
  (`ELEMENTWISE`, `REDUCTION`, `GEMM`, `RESHAPE/VIEW`, `INDEXING`, `BARRIER`)
* [x] Conservative fusion barriers defined (IO, random, sort/select, unsafe multi-output)
* [x] Numerically stable softmax canonicalization
* [x] `softmax` eliminated from IR and lowered into primitive ops
* [x] Structural + numerical regression tests for canonicalization

---

# Fusion (Compiler-Owned)

**Legality phase**

* [x] Legality-only fusion region discovery
* [x] Deterministic toposort-based region construction
* [x] SSA single-consumer enforcement
* [x] Reduction boundary enforcement
* [x] Barrier enforcement

**Cost-gated fusion**

* [x] Greedy forward fusion
* [x] Incremental cost gating (`Δbenefit > Δpenalty`)
* [x] Peak live-byte estimation
* [x] Hard footprint caps for pathological regions
* [x] Cost model validated against large-tensor and over-fusion cases

---

# Kernel Interface Layer (Compiler ↔ Backend Boundary)

* [x] Explicit `FusionRegion` abstraction

* [x] Stable per-region input/output contract

* [x] Region signature determinism (ops, order, shapes, dtypes)

* [x] **KernelIR defined** (lower than GraphIR, higher than MLX)

* [x] **Explicit kernel ABI**

  * inputs
  * outputs
  * temporaries

* [x] Deterministic lowering: `FusionRegion → KernelIR`

* [x] KernelIR structural validator

* [x] Reference (correctness-first) KernelIR executor

* [x] KernelIR → MLX codegen path

* [x] Reference executor ≡ MLX codegen equivalence tests

* [x] Multi-output kernel support (e.g. `meshgrid`)

---

# AI-Generated Kernel Layer (Future, Isolated)

> **This layer does not replace the compiler.
> It plugs in strictly below KernelIR.**

* [ ] Serialize `KernelIR` / `FusionRegion` → prompt-safe representation
* [ ] Constrain generation to:

  * pure MLX
  * no side effects
  * no graph mutation
  * explicit inputs/outputs only
* [ ] Generate candidate kernels for a single region
* [ ] Enforce shape/dtype guards in generated code
* [ ] Static validation of generated kernel vs KernelIR ABI
* [ ] Run correctness check vs KernelIR reference executor

**Optional optimization**

* [ ] Generate multiple variants per region
* [ ] Score variants via:

  * cost model estimate
  * microbenchmark timing
* [ ] Deterministic winner selection

---

# Caching & Reuse

* [ ] Cache generated kernels by:

  * `(region signature, shapes, dtypes, device)`
* [ ] Reuse kernels across graphs and runs
* [ ] Cache invalidation on compiler / KernelIR version change

---

# Execution & Integration

* [x] Deterministic fallback execution for fused regions
* [ ] Dispatch AI-generated kernel when available
* [ ] Fallback to KernelIR interpreter on failure
* [ ] Preserve debuggability:

  * region → kernel → source ops mapping

---

# Validation & Benchmarking

* [x] Fusion legality regression tests

* [x] Cost-model regression tests

* [x] KernelIR lowering / validation / execution tests

* [ ] AI-generated kernel correctness tests

* [ ] Numerical stability stress tests

* [ ] Performance benchmarks:

  * unfused vs fused
  * fused vs AI-generated
  * cold vs cached kernels
