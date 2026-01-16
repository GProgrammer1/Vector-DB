"""Service layer for vector database operations."""

from .indexing_service import IndexingService
from .storage_service import StorageService
from .embedding_client import EmbeddingClient

try:
    from .embedding_service import EmbeddingService
    __all__ = ["IndexingService", "EmbeddingService", "StorageService", "EmbeddingClient"]
except ImportError:
    EmbeddingService = None  # type: ignore[assignment, misc]
    __all__ = ["IndexingService", "StorageService", "EmbeddingClient"]

