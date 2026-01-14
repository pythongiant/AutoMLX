from graph.graph_capture import from_trace
from fusion.regions import find_regions
from fusion.templates import try_template_match

def compile_from_trace(trace_ctx):
    g = from_trace(trace_ctx)
    regions = find_regions(g)
    regions = try_template_match(g, regions)
    return g, regions
