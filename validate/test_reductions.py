# validate/test_reductions.py
import mlx.core as mx
from validate.utils import run_and_compare


def test_sum_mean():
    x = mx.random.normal((4, 6))

    run_and_compare(lambda x: mx.sum(x, axis=1), [x])
    run_and_compare(lambda x: mx.mean(x, axis=0), [x])


def test_softmax_axis():
    x = mx.random.normal((3, 7))

    run_and_compare(lambda x: mx.softmax(x, axis=1), [x])
