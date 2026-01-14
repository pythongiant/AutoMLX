# graph/ir.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class TensorIR:
    tid: int
    shape: Tuple[int, ...]
    dtype: Any
    producer: Optional[int]
    consumers: List[int]

@dataclass
class OpIR:
    op: str
    inputs: List[int]
    outputs: List[int]
    attrs: Dict[str, Any]
    const_args: List[Any] = field(default_factory=list)

@dataclass
class GraphIR:
    ops: List[OpIR]
    tensors: Dict[int, TensorIR]
    outputs: List[int]
