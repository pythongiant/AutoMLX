import weakref
from typing import Optional, List, Tuple


class TensorInfo:
    """
    Graph-level representation of a tensor (SSA value).
    This is NOT an MLX tensor.
    """
    def __init__(
        self,
        tid: int,
        shape: Tuple[int, ...],
        dtype,
        device=None,
    ):
        self.id = tid
        self.shape = shape
        self.dtype = dtype
        self.device = device

        # graph links
        self.producer: Optional[int] = None   # op index
        self.consumers: List[int] = []


class OpNode:
    """
    Graph-level representation of an operation.
    """
    def __init__(
        self,
        op: str,
        inputs: List[int],
        outputs: List[int],
        attrs: Optional[dict] = None,
    ):
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs or {}

class TraceContext:
    enabled = False
    next_tid = 0

    # id(mx.array) -> tid
    tensor_ids = {}

    # tid -> TensorInfo
    tensors = {}

    ops = []


def reset_trace():
    TraceContext.next_tid = 0
    TraceContext.tensor_ids.clear()
    TraceContext.tensors.clear()
    TraceContext.ops.clear()
    
def get_tid(t):
    ctx = TraceContext
    key = id(t)

    if key not in ctx.tensor_ids:
        tid = ctx.next_tid
        ctx.next_tid += 1

        ctx.tensor_ids[key] = tid
        ctx.tensors[tid] = TensorInfo(
            tid=tid,
            shape=tuple(t.shape),
            dtype=t.dtype,
            device=getattr(t, "device", None),
        )

    return ctx.tensor_ids[key]
def new_tid_for_output(t):
    ctx = TraceContext
    tid = ctx.next_tid
    ctx.next_tid += 1

    ctx.tensor_ids[id(t)] = tid
    ctx.tensors[tid] = TensorInfo(
        tid=tid,
        shape=tuple(t.shape),
        dtype=t.dtype,
        device=getattr(t, "device", None),
    )
    return tid
