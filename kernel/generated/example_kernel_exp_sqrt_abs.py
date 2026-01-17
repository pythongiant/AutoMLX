# generated/example_kernel_exp_sqrt_abs.py
import mlx.core as mx

def kernel(t0):
    """
    Hand-written generated kernel.

    Signature:
        input:  t0  (float32, shape preserved)
        output: t1
    """
    t1 = mx.abs(t0)
    t2 = t1 + 1
    t3 = mx.sqrt(t2)
    t4 = mx.exp(t3)
    return t4
