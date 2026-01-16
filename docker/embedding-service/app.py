"""Standalone embedding service API with mmap IPC support."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import os
import asyncio
import threading
import time
import json
import numpy as np
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from vector_db.services.embedding_service import EmbeddingService
from vector_db.services.mmap_ipc import MMapIPC, MessageType

CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.yaml")
MMAP_SHARED_DIR = os.getenv("MMAP_SHARED_DIR", "/app/shared")
embedding_service: Optional[EmbeddingService] = None
mmap_ipc: Optional[MMapIPC] = None
_ipc_running = False

# Rate limit configuration (requests per minute)
RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "200/minute")
RATE_LIMIT_EMBED = os.getenv("RATE_LIMIT_EMBED", "100/minute")
RATE_LIMIT_BATCH = os.getenv("RATE_LIMIT_BATCH", "50/minute")


def process_mmap_requests():
    """Background thread to process mmap IPC requests."""
    global embedding_service, mmap_ipc, _ipc_running
    
    _ipc_running = True
    print(f"Starting mmap IPC request processor (shared_dir: {MMAP_SHARED_DIR})")
    
    while _ipc_running:
        try:
            requests = mmap_ipc.get_pending_requests()
            for slot, msg_type, msg_id, data in requests:
                try:
                    request_data = json.loads(data.decode('utf-8'))
                    
                    if msg_type == MessageType.EMBED_REQUEST:
                        text = request_data["text"]
                        embedding = embedding_service.embed_text(text)
                        mmap_ipc.send_embed_response(msg_id, embedding)
                        # Clear request slot after processing
                        mmap_ipc._clear_slot(mmap_ipc.request_file, slot)
                    elif msg_type == MessageType.BATCH_EMBED_REQUEST:
                        texts = request_data["texts"]
                        embeddings = embedding_service.embed_texts(texts)
                        # Send batch response
                        mmap_ipc.send_batch_embed_response(msg_id, embeddings)
                        # Clear request slot after processing
                        mmap_ipc._clear_slot(mmap_ipc.request_file, slot)
                except Exception as e:
                    error_msg = str(e)
                    mmap_ipc.send_embed_response(msg_id, embedding=None, error=error_msg)
                    # Clear request slot on error
                    mmap_ipc._clear_slot(mmap_ipc.request_file, slot)
                    print(f"Error processing mmap request {msg_id}: {error_msg}")
            
            # Small delay to avoid busy waiting
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in mmap IPC processor: {e}")
            time.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, mmap_ipc, _ipc_running
    
    print(f"Lifespan startup: checking config at {CONFIG_PATH}")
    if not Path(CONFIG_PATH).exists():
        print(f"ERROR: Config file not found at {CONFIG_PATH}")
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    print("Loading embedding model...")
    embedding_service = EmbeddingService(config_path=CONFIG_PATH)
    print(f"Embedding service initialized successfully (dim={embedding_service.dim})")
    
    # Initialize mmap IPC for server role
    mmap_ipc = MMapIPC(MMAP_SHARED_DIR, role="server")
    print(f"Mmap IPC initialized (shared_dir: {MMAP_SHARED_DIR})")
    
    # Start background thread to process mmap requests
    ipc_thread = threading.Thread(target=process_mmap_requests, daemon=True)
    ipc_thread.start()
    print("Mmap IPC request processor started")
    
    yield
    
    # Stop IPC processor
    _ipc_running = False
    

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
@limiter.limit(RATE_LIMIT_EMBED)
def embed_text(http_request: Request, request: EmbedRequest):
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
@limiter.limit(RATE_LIMIT_BATCH)
def embed_texts(http_request: Request, request: EmbedBatchRequest):
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
@limiter.limit(RATE_LIMIT_HEALTH)
def health(request: Request):
    return {"status": "healthy", "service": "embedding"}

