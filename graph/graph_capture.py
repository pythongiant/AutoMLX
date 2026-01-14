# graph/graph_capture.py
from graph.ir import GraphIR, OpIR, TensorIR

def from_trace(trace_ctx) -> GraphIR:
    ops = []
    for op in trace_ctx.ops:
        ops.append(
            OpIR(
                op=op.op,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs=dict(op.attrs or {}),
                const_args=list(op.const_args) if hasattr(op, "const_args") else [],
            )
        )

    tensors = {}
    for tid, t in trace_ctx.tensors.items():
        tensors[tid] = TensorIR(
            tid=tid,
            shape=tuple(t.shape),
            dtype=t.dtype,
            producer=t.producer,
            consumers=list(t.consumers),
        )

    outputs = [
        tid for tid, t in tensors.items()
        if t.producer is not None and len(t.consumers) == 0
    ]

    return GraphIR(
        ops=ops,
        tensors=tensors,
        outputs=outputs,
    )
