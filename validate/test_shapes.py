# validate/test_shapes.py
import mlx.core as mx
from validate.utils import run_and_compare


def test_reshape_transpose():
    x = mx.random.normal((2, 3, 4))

    run_and_compare(
        lambda x: mx.transpose(mx.reshape(x, (4, 6))),
        [x],
    )


def test_broadcast():
    x = mx.random.normal((4, 8))
    b = mx.random.normal((8,))

    run_and_compare(lambda x, b: x + b, [x, b])
