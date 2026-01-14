# validate/test_linear_algebra.py
import mlx.core as mx
from validate.utils import run_and_compare


def test_matmul():
    x = mx.random.normal((3, 4))
    w = mx.random.normal((4, 5))

    run_and_compare(lambda x, w: mx.matmul(x, w), [x, w])


def test_addmm():
    x = mx.random.normal((3, 4))
    w = mx.random.normal((4, 5))
    b = mx.random.normal((5,))

    run_and_compare(lambda x, w, b: mx.addmm(b, x, w), [x, w, b])
