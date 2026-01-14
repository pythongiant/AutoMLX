# runtime/execute.py
import mlx.core as mx
from graph.ir import GraphIR
from fusion.regions import FusionRegion

def run_graph(g: GraphIR, inputs: dict[int, mx.array]) -> dict[int, mx.array]:
    env = dict(inputs)

    for op in g.ops:
        fn = getattr(mx, op.op)

        # rebuild positional args
        args = []
        tensor_iter = iter(op.inputs)

        for a in op.const_args:
            if a is None:
                tid = next(tensor_iter)
                args.append(env[tid])
            else:
                args.append(a)

        out = fn(*args, **(op.attrs or {}))

        if isinstance(out, mx.array):
            env[op.outputs[0]] = out
        elif isinstance(out, (tuple, list)):
            for tid, o in zip(op.outputs, out):
                env[tid] = o
        else:
            raise RuntimeError(f"Unsupported output type from {op.op}")

    return env

