from graph.graph_capture import from_trace
from fusion.regions import find_regions
from fusion.templates import try_template_match
from tracing.tracing import enable_tracing, disable_tracing
from tracing.tensors import TraceContext
from compile.passes.canonicalize_softmax import canonicalize_softmax

def compile_from_trace(trace_ctx):
    g = from_trace(trace_ctx)
    regions = find_regions(g)
    regions = try_template_match(g, regions)
    return g, regions



def trace_and_compile(fn, inputs):
    """
    End-to-end front-end compiler entry point.

    Phases:
      1. Trace eager execution
      2. Build GraphIR (SSA)
      3. Canonicalize graph (expose patterns)
      4. Discover fusion regions (legality-only)
      5. Return enriched GraphIR
    """

    # ------------------------------------------------------------------
    # 1. Trace
    # ------------------------------------------------------------------
    enable_tracing()
    fn(*inputs)

    # Capture input tensor IDs *before* disabling tracing
    input_tids = {}
    for arr in inputs:
        input_tids[id(arr)] = TraceContext.tensor_ids[id(arr)]

    disable_tracing()

    # ------------------------------------------------------------------
    # 2. Graph construction
    # ------------------------------------------------------------------
    g = from_trace(TraceContext)

    # Deterministic input ordering
    g.inputs = list(input_tids.values())

    # ------------------------------------------------------------------
    # 3. Canonicalization passes
    # ------------------------------------------------------------------
    # Example:
    #   softmax(x) → max → sub → exp → sum → div
    g = canonicalize_softmax(g)

    # ------------------------------------------------------------------
    # 4. Fusion region discovery (no cost model yet)
    # ------------------------------------------------------------------
    g.regions = find_regions(g)

    # ------------------------------------------------------------------
    # 5. Return compiled graph
    # ------------------------------------------------------------------
    return g