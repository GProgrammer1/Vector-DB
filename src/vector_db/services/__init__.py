"""Service layer for vector database operations."""

from .indexing_service import IndexingService
from .storage_service import StorageService
from .embedding_client import SyncEmbeddingClient, EmbeddingClient

try:
    from .embedding_service import EmbeddingService
    __all__ = ["IndexingService", "EmbeddingService", "StorageService", "SyncEmbeddingClient", "EmbeddingClient"]
except ImportError:
    EmbeddingService = None  # type: ignore[assignment, misc]
    __all__ = ["IndexingService", "StorageService", "SyncEmbeddingClient", "EmbeddingClient"]

