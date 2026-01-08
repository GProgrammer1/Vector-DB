from pydantic import BaseModel
from typing import Optional, Any


class InsertRequest(BaseModel):
    content: str
    metadata: dict[str, Any] | None = None


class InsertResponse(BaseModel):
    status_code: int
    message: str
    error: Optional[str] = None
