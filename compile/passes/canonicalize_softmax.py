# compile/passes/canonicalize_softmax.py

from graph.ir import GraphIR, OpIR, TensorIR
from graph.op_registry import get_op_kind, OpKind


def canonicalize_softmax(g: GraphIR) -> GraphIR:
    """
    Rewrite:
        softmax(x, axis=a)
    into:
        m = max(x, axis=a, keepdims=True)
        y = exp(x - m)
        s = sum(y, axis=a, keepdims=True)
        out = y / s

    Numerical stability preserved.
    """

    new_ops = []
    next_tid = max(g.tensors.keys(), default=-1) + 1

    def new_tensor_like(ref: TensorIR):
        nonlocal next_tid
        t = TensorIR(
            tid=next_tid,
            shape=ref.shape,
            dtype=ref.dtype,
            producer=None,
            consumers=[],
        )
        g.tensors[next_tid] = t
        next_tid += 1
        return t

    for op_idx, op in enumerate(g.ops):
        if op.op != "softmax":
            new_ops.append(op)
            continue

        # ---- extract inputs ----
        x_tid = op.inputs[0]
        x = g.tensors[x_tid]

        axis = None
        if op.attrs and "axis" in op.attrs:
            axis = op.attrs["axis"]

        # ---- max(x) ----
        m = new_tensor_like(x)
        max_op = OpIR(
            op="max",
            inputs=[x_tid],
            outputs=[m.tid],
            attrs={"axis": axis, "keepdims": True},
            const_args=[None],
        )
        m.producer = len(new_ops)
        new_ops.append(max_op)

        # ---- x - m ----
        xm = new_tensor_like(x)
        sub_op = OpIR(
            op="subtract",
            inputs=[x_tid, m.tid],
            outputs=[xm.tid],
            attrs={},
            const_args=[None, None],
        )
        xm.producer = len(new_ops)
        new_ops.append(sub_op)

        # ---- exp(x - m) ----
        y = new_tensor_like(x)
        exp_op = OpIR(
            op="exp",
            inputs=[xm.tid],
            outputs=[y.tid],
            attrs={},
            const_args=[None],
        )
        y.producer = len(new_ops)
        new_ops.append(exp_op)

        # ---- sum(y) ----
        s = new_tensor_like(x)
        sum_op = OpIR(
            op="sum",
            inputs=[y.tid],
            outputs=[s.tid],
            attrs={"axis": axis, "keepdims": True},
            const_args=[None],
        )
        s.producer = len(new_ops)
        new_ops.append(sum_op)

        # ---- y / s ----
        out = g.tensors[op.outputs[0]]
        div_op = OpIR(
            op="divide",
            inputs=[y.tid, s.tid],
            outputs=[out.tid],
            attrs={},
            const_args=[None, None],
        )
        out.producer = len(new_ops)
        new_ops.append(div_op)

    g.ops = new_ops
    return g
