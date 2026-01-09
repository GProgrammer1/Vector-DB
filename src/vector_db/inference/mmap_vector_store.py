import numpy as np
import random
import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from ..indexing.hnsw import HNSW
from ..storage import MMapNodeStorage
from ..types import Node


class MemoryMappingService:
    """
    Memory mapping service for storing embeddings with integrated indexing.
    
    Uses two-layer memmap storage:
    - Layer 1: Embeddings + IDs (memmap)
    - Layer 2: IDs + Content + Metadata (memmap)
    
    Integrates with HNSW index that persists to disk.
    """

    def __init__(
        self,
        file_path: str,
        dim: int,
        capacity: int,
        config_path: Optional[str] = None,
        index_file: Optional[str] = None,
    ):
        """
        Args:
            file_path: Base path for storage files (will create embedding and metadata files)
            dim: Dimension of embeddings
            capacity: Maximum number of embeddings to store
            config_path: Path to config.yaml for index parameters
            index_file: Path to save/load HNSW index state
        """
        if dim <= 0:
            raise ValueError("Dimension must be greater than 0")
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")

        self.file_path = Path(file_path)
        self.dim = dim
        self.capacity = capacity
        self.size = 0

        # Load index config from file
        if config_path is None:
            raise ValueError("config_path is required")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        index_config = config.get("index", {})
        M = index_config.get("M", 16)
        ef_construction = index_config.get("ef_construction", 200)

        # Initialize two-layer memmap storage
        embedding_file = self.file_path.with_suffix(".embeddings.npy")
        metadata_file = self.file_path.with_suffix(".metadata.npy")
        
        self.storage = MMapNodeStorage(
            embedding_file=embedding_file,
            metadata_file=metadata_file,
            dim=dim,
            capacity=capacity,
        )

        # Initialize HNSW index with persistence
        rng = random.Random(42)
        index_path = Path(index_file) if index_file else self.file_path.with_suffix(".index.pkl")
        self.index = HNSW(
            M=M,
            ef_construction=ef_construction,
            rng=rng,
            storage=self.storage,
            index_file=index_path,
        )

        # Update size from storage
        self.size = self.storage.size()

    def write(
        self,
        embedding: np.ndarray,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Write an embedding to storage and add to index.

        Args:
            embedding: Embedding vector (1D numpy array)
            content: Original text content (optional)
            metadata: Additional metadata dictionary (optional)

        Returns:
            Node ID (index) of the stored embedding
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if embedding.ndim != 1:
            raise ValueError("Embedding must be a 1D array")
        if embedding.size != self.dim:
            raise ValueError(f"Embedding must be of dimension {self.dim}")

        # Get next available node ID
        node_id = self.storage.get_next_id()

        # Create node
        node = Node(
            id=node_id,
            embedding=embedding,
            content=content,
            metadata=metadata or {},
        )

        # Save to storage and add to index
        self.storage.save(node)
        self.index.insert_node(node)

        self.size = self.storage.size()
        return node_id

    def read(self, node_id: int) -> Node:
        """
        Read a node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            Node object with embedding, content, and metadata
        """
        if not isinstance(node_id, int):
            raise TypeError("Node ID must be an integer")
        
        node = self.storage.get(node_id)
        if node is None:
            raise IndexError(f"Node {node_id} not found")
        return node

    def get_embedding(self, node_id: int) -> np.ndarray:
        """Get embedding for a node by ID."""
        return self.storage.get_embedding(node_id)

    def delete(self, node_id: int) -> None:
        """
        Delete a node from storage and index.

        Args:
            node_id: Node ID to delete
        """
        # Delete from index (removes from graph structure)
        self.index.delete_node(node_id)
        
        # Delete from storage
        if hasattr(self.storage, "delete"):
            self.storage.delete(node_id)
        
        self.size = self.storage.size()

    def search(self, query: np.ndarray, k: int, ef: int = 50) -> List[Tuple[Node, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query embedding vector
            k: Number of results to return
            ef: Search width parameter

        Returns:
            List of (Node, distance) tuples
        """
        return self.index.search(query, k=k, ef=ef)
