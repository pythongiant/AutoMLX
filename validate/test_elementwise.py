# validate/test_elementwise.py
import mlx.core as mx
from validate.utils import run_and_compare


def test_unary_ops():
    x = mx.random.normal((5, 5))

    run_and_compare(lambda x: mx.exp(x), [x])
    run_and_compare(lambda x: mx.log(mx.abs(x) + 1), [x])
    run_and_compare(lambda x: mx.tanh(x), [x])


def test_binary_ops():
    x = mx.random.normal((5, 5))
    y = mx.random.normal((5, 5))

    run_and_compare(lambda x, y: x * y + x / (y + 1), [x, y])
