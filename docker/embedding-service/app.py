"""Standalone embedding service API."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from vector_db.services.embedding_service import EmbeddingService

# Initialize embedding service
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.yaml")
embedding_service: Optional[EmbeddingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage embedding service lifecycle: initialize on startup, cleanup on shutdown."""
    global embedding_service
    
    # Initialize embedding service on startup
    if not Path(CONFIG_PATH).exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    embedding_service = EmbeddingService(config_path=CONFIG_PATH)
    print(f"Embedding service initialized (dim={embedding_service.dim})")
    
    yield
    

app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int


class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int


@app.post("/embed", response_model=EmbedResponse)
def embed_text(request: EmbedRequest):
    """Embed a single text."""
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        embedding = embedding_service.embed_text(request.text)
        return EmbedResponse(
            embedding=embedding.tolist(),
            dimension=embedding.shape[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=EmbedBatchResponse)
def embed_texts(request: EmbedBatchRequest):
    """Embed multiple texts."""
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        embeddings = embedding_service.embed_texts(request.texts)
        return EmbedBatchResponse(
            embeddings=[emb.tolist() for emb in embeddings],
            dimension=embeddings.shape[1],
            count=embeddings.shape[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "embedding"}

