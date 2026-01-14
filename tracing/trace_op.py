from tracing.tensors import *
from tracing.tensors import TraceContext, get_tid ,OpNode
import mlx.core as mx

def make_traced_op(name, fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)

        if not TraceContext.enabled:
            return out

        # inputs
        in_tids = []
        for a in args:
            if isinstance(a, mx.array):
                in_tids.append(get_tid(a))
        
        # outputs — ALWAYS fresh
        out_tids = []
        if isinstance(out, mx.array):
            out_tid = new_tid_for_output(out)
            out_tids.append(out_tid)
        

        # register op
        op = OpNode(
            op=name,
            inputs=in_tids,
            outputs=out_tids,
        )
        TraceContext.ops.append(op)

        # link producer / consumers
        for tid in out_tids:
            TraceContext.tensors[tid].producer = len(TraceContext.ops) - 1

        for tid in in_tids:
            TraceContext.tensors[tid].consumers.append(len(TraceContext.ops) - 1)

        return out

    return wrapped
