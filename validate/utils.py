# validate/utils.py
import mlx.core as mx
from tracing.tracing import enable_tracing, disable_tracing
from tracing.tensors import TraceContext
from graph.graph_capture import from_trace
from runtime.execute import run_graph


def run_and_compare(fn, inputs):
    # ---- eager ----
    eager_out = fn(*inputs)

    # ---- trace ----
    enable_tracing()
    traced_out = fn(*inputs)

    # CAPTURE INPUT TIDS BEFORE DISABLING
    input_tids = {}
    for arr in inputs:
        input_tids[arr] = TraceContext.tensor_ids[id(arr)]

    disable_tracing()

    g = from_trace(TraceContext)

    # ---- bind inputs by captured tids ----
    env_inputs = {tid: arr for arr, tid in input_tids.items()}

    env = run_graph(g, env_inputs)
    graph_out = env[g.outputs[0]]

    assert mx.allclose(eager_out, graph_out), "Mismatch eager vs graph"
