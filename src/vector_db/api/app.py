import numpy as np
import yaml
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Set
from fastapi import FastAPI, HTTPException

from .models import InsertRequest, InsertResponse
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



EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")

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
        
    if not EMBEDDING_SERVICE_URL:
        raise ValueError("EMBEDDING_SERVICE_URL environment variable is required")
    embedding_client = EmbeddingClient(base_url=EMBEDDING_SERVICE_URL, use_async=False)
    if embedding_client.health_check():
        print(f"Embedding service client initialized (URL: {EMBEDDING_SERVICE_URL})")
    else:
        print(f"Warning: Embedding service at {EMBEDDING_SERVICE_URL} is not healthy (will retry on requests)")

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
    
    if embedding_client and hasattr(embedding_client, 'close'):
        embedding_client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "index_loaded": indexing_service.is_index_loaded() if indexing_service else False,
        "index_size": indexing_service.get_index_size() if indexing_service else 0,
        "storage_size": storage_service.size() if storage_service else 0,
        "index_modified": indexing_service._index_modified if indexing_service else False,
    }


@app.post("/embed", response_model=InsertResponse)
def embed_document(insert_request: InsertRequest):
    global embedding_client, storage_service, indexing_service
    
    if embedding_client is None or storage_service is None or indexing_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    
    try:
        content = insert_request.content
        embedding = embedding_client.embed_text(content)
        
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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


from .models import InsertRequest, InsertResponse, QueryRequest, QueryResponse

@app.post("/search", response_model=QueryResponse)
def search_index(query_request: QueryRequest):
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

    

