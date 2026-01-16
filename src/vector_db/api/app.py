import numpy as np
import yaml
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Set
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .models import InsertRequest, InsertResponse, BatchInsertRequest, BatchInsertResponse, QueryRequest, QueryResponse
from ..services.storage_service import StorageService
from ..services.indexing_service import IndexingService
from ..services.embedding_client import EmbeddingClient
from ..types import Node

def get_config_path() -> str:
    config_path = os.getenv("CONFIG_PATH")
    if not config_path:
        raise ValueError(
            "CONFIG_PATH environment variable is required. "
            "Set it to the path of your config.yaml file."
        )
    return str(Path(config_path).resolve())

CONFIG_PATH: Optional[str] = os.getenv("CONFIG_PATH")
if CONFIG_PATH:
    CONFIG_PATH = str(Path(CONFIG_PATH).resolve())

embedding_client: Optional[EmbeddingClient] = None
storage_service: Optional[StorageService] = None
indexing_service: Optional[IndexingService] = None



MMAP_SHARED_DIR = os.getenv("MMAP_SHARED_DIR", "/tmp/vector_db_shared")

RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "100/minute")
RATE_LIMIT_EMBED = os.getenv("RATE_LIMIT_EMBED", "30/minute")
RATE_LIMIT_SEARCH = os.getenv("RATE_LIMIT_SEARCH", "60/minute")

def load_config() -> dict:
    config_path = CONFIG_PATH or get_config_path()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_client, storage_service, indexing_service

    if embedding_client is not None and storage_service is not None and indexing_service is not None:
        yield
        return
    
    config = load_config()
    
  
    vector_db_config = config["vector_db"]
    
    embedding_config = config["embedding"]

    file_path = vector_db_config["file_path"]
 
    capacity = vector_db_config["capacity"]

    dim = embedding_config["dimension"]
        
    embedding_client = EmbeddingClient(
        use_async=True,
        mmap_shared_dir=MMAP_SHARED_DIR
    )
    print(f"Embedding client initialized with mmap IPC (shared_dir: {MMAP_SHARED_DIR})")

    storage_service = StorageService(
        file_path=file_path,
        dim=dim,
        capacity=capacity,
    )
    print(f"Storage service initialized (size={storage_service.size()})")
    
    index_file = Path(file_path).with_suffix(".index.pkl")
    indexing_service = IndexingService(
        storage=storage_service.storage,
        config_path=str(CONFIG_PATH),
        index_file=str(index_file),
    )
    
    if indexing_service.is_index_loaded():
        print(f"Index loaded from {index_file} (size={storage_service.size()} nodes)")
    else:
        print(f"Index not found at {index_file}, will create on first insert")
    
    yield
    
    if indexing_service:
        indexing_service.save_index()
        print(f"Index saved to disk on shutdown (size={indexing_service.get_index_size()} nodes)")
    


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
@limiter.limit(RATE_LIMIT_HEALTH)
def health(request: Request):
    return {
        "status": "healthy",
        "index_loaded": indexing_service.is_index_loaded() if indexing_service else False,
        "index_size": indexing_service.get_index_size() if indexing_service else 0,
        "storage_size": storage_service.size() if storage_service else 0,
        "index_modified": indexing_service._index_modified if indexing_service else False,
    }


MAX_DOCUMENT_SIZE = os.getenv("MAX_DOCUMENT_SIZE", 50 * 1024 * 1024)
MAX_DOCUMENT_SIZE = int(MAX_DOCUMENT_SIZE)

@app.post("/embed", response_model=InsertResponse)
@limiter.limit(RATE_LIMIT_EMBED)
async def embed_document(request: Request, insert_request: InsertRequest):
    global embedding_client, storage_service, indexing_service
    
    if embedding_client is None or storage_service is None or indexing_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    
    try:
        content = insert_request.content
        
        # Validate document size (50MB limit)
        content_size = len(content.encode('utf-8'))
        if content_size > MAX_DOCUMENT_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Document size ({content_size / (1024*1024):.2f}MB) exceeds maximum allowed size {MAX_DOCUMENT_SIZE / (1024*1024):.2f}MB"
            )
        
        embedding = await embedding_client.embed_text_async(content)
        
        node_id = storage_service.get_next_id()
        node = Node(
            id=node_id,
            embedding=embedding,
            content=content,
            metadata=insert_request.metadata or {},
        )
        
        storage_service.save(node)
        indexing_service.insert_node(node)
        
        return InsertResponse(
            status_code=200,
            message=f"Document embedded and stored successfully at index {node_id}",
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/embed/batch", response_model=BatchInsertResponse)
@limiter.limit(RATE_LIMIT_EMBED)
async def embed_documents_batch(request: Request, batch_request: BatchInsertRequest):
    global embedding_client, storage_service, indexing_service
    
    if embedding_client is None or storage_service is None or indexing_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    
    try:
        documents = batch_request.documents
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="Empty document list"
            )
        
        # Validate all document sizes
        contents = []
        metadatas = []
        for doc in documents:
            if "content" not in doc:
                raise HTTPException(
                    status_code=400,
                    detail="Each document must have a 'content' field"
                )
            content = doc["content"]
            content_size = len(content.encode('utf-8'))
            if content_size > MAX_DOCUMENT_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Document size ({content_size / (1024*1024):.2f}MB) exceeds maximum allowed size {MAX_DOCUMENT_SIZE / (1024*1024):.2f}MB"
                )
            contents.append(content)
            metadatas.append(doc.get("metadata") or {})
        
        # Batch embed all documents
        embeddings = await embedding_client.embed_texts_async(contents)
        
        # Store all documents and index them
        inserted_ids = []
        for i, (content, embedding, metadata) in enumerate(zip(contents, embeddings, metadatas)):
            node_id = storage_service.get_next_id()
            node = Node(
                id=node_id,
                embedding=embedding,
                content=content,
                metadata=metadata,
            )
            storage_service.save(node)
            indexing_service.insert_node(node)
            inserted_ids.append(node_id)
        
        return BatchInsertResponse(
            status_code=200,
            message=f"Successfully embedded and stored {len(inserted_ids)} documents",
            inserted_count=len(inserted_ids),
            inserted_ids=inserted_ids,
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch request: {str(e)}"
        )


@app.post("/search", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_SEARCH)
def search_index(request: Request, query_request: QueryRequest):
    global embedding_client, storage_service, indexing_service
    
    if embedding_client is None or storage_service is None or indexing_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    
    try:
        query_embedding = embedding_client.embed_text(query_request.query)
        
        filter_ids: Optional[Set[int]] = None
        if query_request.metadata_filter:
            filter_ids = storage_service.filter_by_metadata(query_request.metadata_filter)
            if not filter_ids:
                return QueryResponse(
                    status_code=200,
                    results=[],
                    error=None
                )
        
        search_kwargs: Dict[str, Any] = {
            "filter_ids": filter_ids
        }
        
        if query_request.ef is not None:
            search_kwargs["ef"] = query_request.ef
        if query_request.n_probe is not None:
            search_kwargs["n_probe"] = query_request.n_probe
        if query_request.pq_chunks:
            search_kwargs["pq_chunks"] = query_request.pq_chunks
        if query_request.params:
            search_kwargs.update(query_request.params)
            
        results = indexing_service.search(
            query=query_embedding,
            k=query_request.top_k,
            **search_kwargs
        )
        
        formatted_results = []
        for node, dist in results:
            formatted_results.append({
                "id": node.id,
                "content": node.content,
                "metadata": node.metadata,
                "distance": float(dist)
            })
            
        return QueryResponse(
            status_code=200,
            results=formatted_results,
            error=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search: {str(e)}"
        )

    

