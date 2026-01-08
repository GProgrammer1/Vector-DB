from pydantic import BaseModel
from typing import Optional, Any, Dict


class InsertRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class InsertResponse(BaseModel):
    status_code: int
    message: str
    error: Optional[str] = None
