# tracing/trace_op.py
import mlx.core as mx
from tracing.tensors import TraceContext, OpNode, get_tid, new_tid_for_output

def make_traced_op(name, fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)

        if not TraceContext.enabled:
            return out

        in_tids = []
        const_args = []

        # ---- separate tensor args from constants ----
        for a in args:
            if isinstance(a, mx.array):
                in_tids.append(get_tid(a))
                const_args.append(None)   # placeholder
            else:
                const_args.append(a)      # literal

        # ---- outputs (SSA) ----
        out_tids = []

        def register_output(o):
            tid = new_tid_for_output(o)
            out_tids.append(tid)

        if isinstance(out, mx.array):
            register_output(out)
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, mx.array):
                    register_output(o)

        op_idx = len(TraceContext.ops)
        TraceContext.ops.append(
            OpNode(
                op=name,
                inputs=in_tids,
                outputs=out_tids,
                attrs=kwargs,
                const_args=const_args,
            )
        )

        for tid in out_tids:
            TraceContext.tensors[tid].producer = op_idx

        for tid in in_tids:
            TraceContext.tensors[tid].consumers.append(op_idx)

        return out

    return wrapped
