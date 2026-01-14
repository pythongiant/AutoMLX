# fusion/templates.py
from typing import List
from graph.ir import GraphIR
from fusion.regions import FusionRegion

def try_template_match(g: GraphIR, regions: List[FusionRegion]) -> List[FusionRegion]:
    """
    Upgrade certain regions to template kernels.
    Keep it simple: only local within region for now.
    """
    out: List[FusionRegion] = []
    for r in regions:
        if r.kind != "generic":
            out.append(r)
            continue

        ops = [g.ops[i].op for i in r.op_indices]

        # Example template: add -> softmax (common post-op fusion)
        if ops == ["add", "softmax"]:
            r.kind = "template"
            r.template = "fused_add_softmax"
            out.append(r)
            continue

        # Later: recognize attention blocks, matmul+bias+gelu, etc.
        out.append(r)

    return out
