# kernel/codegen_mlx.py
"""
KernelIR → MLX executable lowering.

This produces a fast Python callable backed by MLX ops.
"""

import mlx.core as mx
from kernel.ir import KernelIR


def codegen_mlx(kernel: KernelIR):
    """
    Lower KernelIR to an executable MLX callable.

    Returns:
        fn(*inputs) -> output or tuple(outputs)
    """

    # Pre-resolve MLX functions
    ops = []
    for op in kernel.ops:
        try:
            fn = getattr(mx, op.op)
        except AttributeError:
            raise NotImplementedError(f"Unsupported op in MLX codegen: {op.op}")

        ops.append((fn, op.inputs, op.outputs, op.attrs or {}))

    input_tids = kernel.inputs
    output_tids = kernel.outputs

    # Precompute env size
    max_tid = max(
        input_tids + output_tids +
        [tid for _, ins, outs, _ in ops for tid in (*ins, *outs)]
    )
    env_size = max_tid + 1

    def kernel_fn(*args):
        assert len(args) == len(input_tids)

        env = [None] * env_size

        # bind inputs
        for tid, val in zip(input_tids, args):
            env[tid] = val

        # execute
        for fn, ins, outs, attrs in ops:
            vals = [env[i] for i in ins]
            res = fn(*vals, **attrs)

            if len(outs) == 1:
                env[outs[0]] = res
            else:
                for tid, v in zip(outs, res):
                    env[tid] = v

        # return outputs
        if len(output_tids) == 1:
            return env[output_tids[0]]
        return tuple(env[tid] for tid in output_tids)

    return kernel_fn
