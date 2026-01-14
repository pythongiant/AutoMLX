# validate/test_canonicalize_softmax.py

import mlx.core as mx

from compile.pipeline import trace_and_compile
from runtime.execute import run_graph


def test_softmax_is_eliminated_and_correct():
    """
    Canonicalization test:
    - softmax op must be eliminated from GraphIR
    - rewritten graph must contain max, sum, exp, subtract, divide
    - numerical output must match eager softmax
    """

    x = mx.random.normal((4, 8))

    # ---- compile ----
    g = trace_and_compile(lambda x: mx.softmax(x, axis=1), [x])

    # ---- structural assertions ----
    ops = [op.op for op in g.ops]

    assert "softmax" not in ops, "softmax should be eliminated by canonicalization"
    assert "max" in ops, "max reduction missing in softmax canonicalization"
    assert "sum" in ops, "sum reduction missing in softmax canonicalization"
    assert "exp" in ops, "exp missing in softmax canonicalization"
    assert "subtract" in ops, "subtract missing in softmax canonicalization"
    assert "divide" in ops, "divide missing in softmax canonicalization"

    # exactly one divide at the end (y / sum)
    assert ops.count("divide") == 1

    # ---- execute compiled graph ----
    env = run_graph(g, {g.inputs[0]: x})
    compiled_out = env[g.outputs[0]]

    # ---- eager reference ----
    eager_out = mx.softmax(x, axis=1)

    # ---- numerical correctness ----
    assert mx.allclose(compiled_out, eager_out, atol=1e-6)


def test_softmax_attention_exposure_pattern():
    """
    Ensures canonicalization exposes attention-style pattern:
        matmul -> add -> max -> sub -> exp -> sum -> div
    """

    x = mx.random.normal((2, 4))
    w = mx.random.normal((4, 6))
    b = mx.random.normal((6,))

    def fn(x, w, b):
        return mx.softmax(mx.matmul(x, w) + b, axis=-1)

    g = trace_and_compile(fn, [x, w, b])

    ops = [op.op for op in g.ops]

    # ---- no softmax remains ----
    assert "softmax" not in ops

    # ---- attention-relevant ops present ----
    assert "matmul" in ops
    assert "add" in ops
    assert "max" in ops
    assert "exp" in ops
    assert "sum" in ops
    assert "divide" in ops

    # ---- execute + compare ----
    env = run_graph(
        g,
        {
            g.inputs[0]: x,
            g.inputs[1]: w,
            g.inputs[2]: b,
        },
    )

    compiled_out = env[g.outputs[0]]
    eager_out = fn(x, w, b)

    assert mx.allclose(compiled_out, eager_out, atol=1e-6)
