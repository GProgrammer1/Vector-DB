"""Storage service for managing node storage lifecycle."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Set, List

from ..storage import MMapNodeStorage
from ..types import Node


class StorageService:
    """
    Service for managing node storage lifecycle.
    
    Handles:
    - Initializing memmap storage
    - Node CRUD operations
    - Storage size tracking
    """

    def __init__(
        self,
        file_path: str,
        dim: int,
        capacity: int,
    ):
        """
        Initialize the storage service.
        
        Args:
            file_path: Base path for storage files
            dim: Dimension of embeddings
            capacity: Maximum number of embeddings to store
        """
        if dim <= 0:
            raise ValueError("Dimension must be greater than 0")
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")

        self.file_path = Path(file_path)
        self.dim = dim
        self.capacity = capacity

        # Initialize two-layer memmap storage
        embedding_file = self.file_path.with_suffix(".embeddings.npy")
        metadata_file = self.file_path.with_suffix(".metadata.npy")
        
        self._storage = MMapNodeStorage(
            embedding_file=embedding_file,
            metadata_file=metadata_file,
            dim=dim,
            capacity=capacity,
        )

    def save(self, node: Node) -> None:
        """
        Save a node to storage.
        
        Args:
            node: Node to save
        """
        self._storage.save(node)

    def get(self, node_id: int) -> Optional[Node]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Node if found, None otherwise
        """
        return self._storage.get(node_id)

    def get_embedding(self, node_id: int) -> np.ndarray:
        """
        Get embedding for a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Embedding vector
        """
        return self._storage.get_embedding(node_id)

    def delete(self, node_id: int) -> None:
        """
        Delete a node from storage.
        
        Args:
            node_id: Node ID to delete
        """
        self._storage.delete(node_id)

    def get_next_id(self) -> int:
        """
        Get the next available node ID.
        
        Returns:
            Next node ID
        """
        return self._storage.get_next_id()

    def filter_by_metadata(self, filter_dict: Dict[str, Any]) -> Set[int]:
        """
        Filter nodes by metadata.
        
        Args:
            filter_dict: Dictionary of key-value pairs to match
            
        Returns:
            Set of node IDs that match the filter
        """
        matching_ids = set()
        # Full scan - can be slow for very large datasets
        for i in range(self._storage.get_next_id()):
            node = self._storage.get(i)
            if node:
                match = True
                for k, v in filter_dict.items():
                    if node.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    matching_ids.add(i)
        return matching_ids

    def size(self) -> int:
        """
        Get current number of stored nodes.
        
        Returns:
            Number of nodes in storage
        """
        return self._storage.size()

    @property
    def storage(self) -> MMapNodeStorage:
        """Get the underlying storage instance (for indexing service)."""
        return self._storage

