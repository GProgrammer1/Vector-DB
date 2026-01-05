"""Inference module for vector database operations."""

from vector_db.inference.device import (
    DeviceType,
    get_device,
    get_device_info,
    is_gpu_available,
)
from vector_db.inference.mmap_vector_store import MemoryMappingService
from vector_db.inference.pq import ProductQuantizationService

# Optional imports - may not be available if dependencies aren't installed
try:
    from vector_db.inference.embedding import EmbeddingService
except ImportError:
    EmbeddingService = None  # type: ignore[assignment, misc]

__all__ = [
    "DeviceType",
    "get_device",
    "get_device_info",
    "is_gpu_available",
    "MemoryMappingService",
    "ProductQuantizationService",
]

if EmbeddingService is not None:
    __all__.append("EmbeddingService")

