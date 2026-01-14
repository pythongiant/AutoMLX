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

    Steps:
      1. Enable tracing
      2. Execute function eagerly (records trace)
      3. Capture GraphIR
      4. Run canonicalization passes
      5. Return compiled GraphIR
    """

    # ---- trace ----
    enable_tracing()
    fn(*inputs)

    # IMPORTANT: capture input tids BEFORE disabling tracing
    input_tids = {}
    for arr in inputs:
        input_tids[id(arr)] = TraceContext.tensor_ids[id(arr)]

    disable_tracing()

    # ---- build graph ----
    g = from_trace(TraceContext)

    # ---- record graph inputs deterministically ----
    g.inputs = list(input_tids.values())

    # ---- canonicalization passes ----
    g = canonicalize_softmax(g)

    return g