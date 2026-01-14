# validate/test_multi_output.py
import mlx.core as mx
from validate.utils import run_and_compare


def test_split():
    x = mx.random.normal((10,))

    def fn(x):
        a, b = mx.split(x, 2)
        return a + b

    run_and_compare(fn, [x])


def test_meshgrid():
    x = mx.arange(3)
    y = mx.arange(4)

    def fn(x, y):
        X, Y = mx.meshgrid(x, y)
        return X + Y

    run_and_compare(fn, [x, y])
