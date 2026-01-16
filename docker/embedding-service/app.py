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

CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.yaml")
embedding_service: Optional[EmbeddingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service
    
    print(f"Lifespan startup: checking config at {CONFIG_PATH}")
    if not Path(CONFIG_PATH).exists():
        print(f"ERROR: Config file not found at {CONFIG_PATH}")
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    print("Loading embedding model...")
    embedding_service = EmbeddingService(config_path=CONFIG_PATH)
    print(f"Embedding service initialized successfully (dim={embedding_service.dim})")
    
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
    return {"status": "healthy", "service": "embedding"}

