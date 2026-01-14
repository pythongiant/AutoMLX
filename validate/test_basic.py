# validate/test_basic.py
import mlx.core as mx
from validate.utils import run_and_compare


def test_add_softmax():
    x = mx.random.normal((4, 8))
    b = mx.random.normal((8,))

    run_and_compare(
        lambda x, b: mx.softmax(x + b),
        [x, b],
    )
