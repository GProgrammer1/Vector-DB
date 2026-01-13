"""Indexing service for managing HNSW index lifecycle."""

import numpy as np
import random
import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set

from ..indexing.hnsw import HNSW
from ..storage import MMapNodeStorage
from ..types import Node


class IndexingService:
    """
    Service for managing HNSW index lifecycle.
    
    Handles:
    - Loading index from disk on initialization (if exists)
    - Saving index to disk after modifications
    - Creating new index if it doesn't exist
    """

    def __init__(
        self,
        storage: MMapNodeStorage,
        config_path: str,
        index_file: Optional[str] = None,
    ):
        """
        Initialize the indexing service.
        
        Args:
            storage: MMapNodeStorage instance for node data
            config_path: Path to config.yaml for index parameters
            index_file: Path to save/load HNSW index state (optional)
        """
        self.storage = storage
        self.config_path = Path(config_path)
        
        # Load index config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        index_config = config.get("index", {})
        M = index_config.get("M", 16)
        ef_construction = index_config.get("ef_construction", 200)
        
        # Determine index file path
        if index_file:
            self.index_file = Path(index_file)
        else:
            # Default: use storage base path
            base_path = Path(storage.embedding_file).parent / Path(storage.embedding_file).stem.replace(".embeddings", "")
            self.index_file = base_path.with_suffix(".index.pkl")
        
        # Initialize HNSW with persistence
        rng = random.Random(42)
        self.index = HNSW(
            M=M,
            ef_construction=ef_construction,
            rng=rng,
            storage=storage,
            index_file=self.index_file,
        )
        
        # Track if index was loaded from disk
        self._index_loaded = self.index_file.exists()
        self._index_modified = False
        
        # Load flush threshold from config
        self.flush_threshold = index_config.get("flush_threshold", 1000)

    def is_index_loaded(self) -> bool:
        """Check if index was loaded from disk."""
        return self._index_loaded

    def insert_node(self, node: Node) -> None:
        """
        Insert a node into the index.
        
        Args:
            node: Node to insert
        """
        self.index.insert_node(node)
        self._index_modified = True
        
        # Check if threshold reached and flush if needed
        if self._should_flush():
            self.save_index()

    def delete_node(self, node_id: int) -> None:
        """
        Delete a node from the index.
        
        Args:
            node_id: ID of node to delete
        """
        self.index.delete_node(node_id)
        self._index_modified = True

    def search(
        self, query: np.ndarray, k: int, **kwargs
    ) -> List[Tuple[Node, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query embedding vector
            k: Number of results to return
            **kwargs: Additional parameters (ef, filter_ids, pq_chunks, etc.)
            
        Returns:
            List of (Node, distance) tuples
        """
        return self.index.search(query, k=k, **kwargs)

    def save_index(self) -> None:
        """Save index to disk if it has been modified."""
        if self._index_modified:
            self.index.save_index()
            self._index_modified = False

    def force_save_index(self) -> None:
        """Force save index to disk even if not modified."""
        self.index.save_index()
        self._index_modified = False

    def get_index_size(self) -> int:
        """
        Get current number of nodes in the index.
        
        Returns:
            Number of nodes in index
        """
        return len(self.index.node_store)

    def _should_flush(self) -> bool:
        """
        Check if index should be flushed to disk based on threshold.
        
        Returns:
            True if threshold reached, False otherwise
        """
        return self.get_index_size() >= self.flush_threshold

