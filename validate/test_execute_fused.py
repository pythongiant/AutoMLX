# validate/test_execute_fused.py

import mlx.core as mx

from compile.pipeline import trace_and_compile
from runtime.execute_fused import run_fused_regions


def test_fused_region_execution_matches_eager():
    x = mx.random.normal((8, 8))

    def fn(x):
        return mx.exp(mx.sqrt(mx.abs(x) + 1))

    g = trace_and_compile(fn, [x])

    env = run_fused_regions(g, g.regions, {g.inputs[0]: x})
    fused_out = env[g.outputs[0]]

    eager_out = fn(x)

    assert mx.allclose(fused_out, eager_out)


def test_fusion_respects_reduction_barrier():
    x = mx.random.normal((4, 6))

    def fn(x):
        y = mx.exp(x)
        z = mx.sum(y, axis=1)
        return z

    g = trace_and_compile(fn, [x])

    # expect multiple regions
    assert len(g.regions) >= 2

    env = run_fused_regions(g, g.regions, {g.inputs[0]: x})
    fused_out = env[g.outputs[0]]

    eager_out = fn(x)

    assert mx.allclose(fused_out, eager_out)
