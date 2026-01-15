from pydantic import BaseModel
from typing import Optional, Any, Dict, List


class InsertRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class InsertResponse(BaseModel):
    status_code: int
    message: str
    error: Optional[str] = None
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None
    pq_chunks: Optional[int] = None  # For PQ-enabled searches
    ef: int = 50  # For HNSW
    n_probe: int = 10  # For IVF
    params: Optional[Dict[str, Any]] = None # For any additional parameters


class QueryResponse(BaseModel):
    status_code: int
    results: List[Dict[str, Any]]
    error: Optional[str] = None
