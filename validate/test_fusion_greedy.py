# test_fusion_greedy.py
#
# Structural tests for greedy fusion.
# These tests validate *fusion decisions*, not just numerical correctness.
# They ensure legality, determinism, and cost-model gating.

import mlx.core as mx

from compile.pipeline import trace_and_compile
from fusion.greedy_fuse import greedy_fuse


def _count_regions(fused):
    return len(fused)


def _ops(fused_region):
    return fused_region.op_indices


# -------------------------------------------------
# Basic fusion: elementwise chain should fuse
# -------------------------------------------------

def test_elementwise_chain_fuses():
    def fn(x):
        return mx.exp(mx.log(mx.sqrt(x)))

    x = mx.random.normal((16, 16))
    g = trace_and_compile(fn, [x])

    fused = greedy_fuse(g)

    # exp(log(sqrt(x))) should be one fused region
    assert _count_regions(fused) == 1
    assert len(_ops(fused[0])) == 3


# -------------------------------------------------
# Barrier prevents fusion
# -------------------------------------------------

def test_barrier_blocks_fusion():
    def fn(x, w):
        y = mx.matmul(x, w)
        return mx.exp(y)

    x = mx.random.normal((8, 8))
    w = mx.random.normal((8, 8))
    g = trace_and_compile(fn, [x, w])

    fused = greedy_fuse(g)

    # matmul must be isolated
    assert _count_regions(fused) == 2
    assert len(_ops(fused[0])) == 1  # matmul
    assert len(_ops(fused[1])) == 1  # exp


# -------------------------------------------------
# Reduction boundary prevents fusion
# -------------------------------------------------

def test_reduction_boundary_blocks_fusion():
    def fn(x):
        y = mx.sum(x, axis=1)
        return mx.exp(y)

    x = mx.random.normal((32, 32))
    g = trace_and_compile(fn, [x])

    fused = greedy_fuse(g)

    # sum and exp should not fuse
    assert _count_regions(fused) == 2
    assert len(_ops(fused[0])) == 1
    assert len(_ops(fused[1])) == 1


# -------------------------------------------------
# Single-consumer invariant enforced
# -------------------------------------------------

def test_single_consumer_required():
    def fn(x):
        y = mx.exp(x)
        z1 = mx.log(y)
        z2 = mx.sqrt(y)
        return z1 + z2

    x = mx.random.normal((16, 16))
    g = trace_and_compile(fn, [x])

    fused = greedy_fuse(g)

    # exp feeds two consumers → must not fuse with either
    assert _count_regions(fused) >= 3
    # exp must be isolated
    assert any(len(r.op_indices) == 1 for r in fused)


# -------------------------------------------------
# Cost model prevents over-fusion
# -------------------------------------------------

def test_cost_model_blocks_large_fusion():
    def fn(x):
        y = x
        for _ in range(8):
            y = mx.exp(y)
        return y

    # Large tensor → footprint penalty dominates
    x = mx.random.normal((1024, 1024))
    g = trace_and_compile(fn, [x])

    fused = greedy_fuse(g)

    # Should not fuse everything into one giant region
    assert _count_regions(fused) > 1


# -------------------------------------------------
# Determinism test
# -------------------------------------------------

def test_fusion_deterministic():
    def fn(x):
        return mx.exp(mx.log(mx.sqrt(x)))

    x = mx.random.normal((32, 32))
    g1 = trace_and_compile(fn, [x])
    g2 = trace_and_compile(fn, [x])

    fused1 = greedy_fuse(g1)
    fused2 = greedy_fuse(g2)

    assert [r.op_indices for r in fused1] == [r.op_indices for r in fused2]
