from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class Node:
    id: int
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
