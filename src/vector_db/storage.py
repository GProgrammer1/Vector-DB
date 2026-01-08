"""Storage abstraction for vector database nodes."""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
from pathlib import Path

import numpy as np

from vector_db.types import Node


class NodeStorage(ABC):
    """Abstract base class for node storage backends."""

    @abstractmethod
    def save(self, node: Node) -> None:
        """Save a node to storage."""
        pass

    @abstractmethod
    def get(self, node_id: int) -> Optional[Node]:
        """Retrieve a node by ID."""
        pass

    @abstractmethod
    def get_embedding(self, node_id: int) -> np.ndarray:
        """Get embedding for a node by ID."""
        pass

    @abstractmethod
    def get_all_ids(self) -> list[int]:
        """Get all node IDs in storage."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get the number of nodes in storage."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage and release resources."""
        pass

    @abstractmethod
    def add(self, node: Node) -> None:
        """Add a node to storage (alias for save for compatibility)."""
        pass


class InMemoryNodeStorage(NodeStorage):
    """In-memory storage backend (mimics current behavior)."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._nodes: dict[int, Node] = {}
        self._next_id = 0

    def save(self, node: Node) -> None:
        """Save a node to memory."""
        self._nodes[node.id] = node
        if node.id >= self._next_id:
            self._next_id = node.id + 1

    def add(self, node: Node) -> None:
        """Add a node to storage (alias for save)."""
        self.save(node)

    def get(self, node_id: int) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def get_embedding(self, node_id: int) -> np.ndarray:
        """Get embedding for a node by ID."""
        node = self.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} not found")
        return node.embedding

    def get_all_ids(self) -> list[int]:
        """Get all node IDs."""
        return list(self._nodes.keys())

    def size(self) -> int:
        """Get the number of nodes."""
        return len(self._nodes)

    def close(self) -> None:
        """Close storage (no-op for in-memory)."""
        pass

    def get_next_id(self) -> int:
        """Get the next available node ID."""
        return self._next_id


class MMapNodeStorage(NodeStorage):
    """
    Two-layer memmap storage for scalable vector database.
    
    Layer 1: Embeddings + IDs (memmap)
    Layer 2: IDs + Content + Metadata (memmap with structured dtype)
    
    This allows efficient disk-backed storage without loading everything into memory.
    """

    def __init__(
        self,
        embedding_file: Union[str, Path],
        metadata_file: Union[str, Path],
        dim: int,
        capacity: int = 1_000_000,
    ):
        """
        Initialize two-layer memmap storage.

        Args:
            embedding_file: Path to memmap file for embeddings + ids
            metadata_file: Path to memmap file for id + content + metadata
            dim: Dimension of embeddings
            capacity: Maximum number of nodes to store
        """
        self.embedding_file = Path(embedding_file)
        self.metadata_file = Path(metadata_file)
        self.dim = dim
        self.capacity = capacity

        # Initialize memmaps
        self._init_embedding_memmap()
        self._init_metadata_memmap()

    def _init_embedding_memmap(self) -> None:
        """Initialize memmap for embeddings and IDs."""
        mode = "r+" if self.embedding_file.exists() else "w+"
        dtype: np.dtype = np.dtype([("id", np.int64), ("embedding", np.float32, (self.dim,))])
        self.embeddings: np.memmap = np.memmap(
            self.embedding_file,
            dtype=dtype,
            mode=mode,
            shape=(self.capacity,),
        )

        # Track size and id_to_index mapping
        self._id_to_index: dict[int, int] = {}
        if mode == "r+":
            # Find all valid entries 
            # We use a heuristic: if embedding has non-zero values, it's valid
            valid_indices = []
            for idx in range(self.capacity):
                node_id = int(self.embeddings[idx]["id"])
                embedding = self.embeddings[idx]["embedding"]
                # Valid if id >= 0 and embedding is not all zeros
                if node_id >= 0 and np.any(embedding != 0.0):
                    valid_indices.append(idx)
                    self._id_to_index[node_id] = idx
            self._size = len(valid_indices)
        else:
            self._size = 0

    def _init_metadata_memmap(self) -> None:
        """Initialize memmap for metadata (id, content, metadata_json)."""
        mode = "r+" if self.metadata_file.exists() else "w+"
       
        max_content_len = 10240  # 10KB
        max_metadata_len = 5120  # 5KB
        
        dtype: np.dtype = np.dtype([
            ("id", np.int64),
            ("content", f"U{max_content_len}"), 
            ("metadata_json", f"U{max_metadata_len}"),
        ])
        
        self.metadata: np.memmap = np.memmap(
            self.metadata_file,
            dtype=dtype,
            mode=mode,
            shape=(self.capacity,),
        )

    def _find_free_slot(self) -> int:
        """Find the next free slot in the memmap."""
        if self._size >= self.capacity:
            raise RuntimeError(f"Storage capacity ({self.capacity}) exceeded")
        
        # Find first slot with id < 0 or embedding is all zeros
        for idx in range(self.capacity):
            node_id = int(self.embeddings[idx]["id"])
            embedding = self.embeddings[idx]["embedding"]
            if node_id < 0 or np.all(embedding == 0.0):
                return idx
        
        # If no free slot found, use size as index
        return self._size

    def save(self, node: Node) -> None:
        """Save a node to memmap storage."""
        # Check if node already exists (update case)
        if node.id in self._id_to_index:
            idx = self._id_to_index[node.id]
        else:
            idx = self._find_free_slot()
            self._id_to_index[node.id] = idx
            self._size += 1

        # Store embedding + id in layer 1
        self.embeddings[idx]["id"] = node.id
        self.embeddings[idx]["embedding"] = node.embedding.astype(np.float32)

        # Store metadata in layer 2 (same index)
        metadata_json = json.dumps(node.metadata) if node.metadata else ""
        content = node.content or ""
        
        # Truncate if too long 
        max_content_chars = self.metadata.dtype["content"].itemsize // 4
        max_metadata_chars = self.metadata.dtype["metadata_json"].itemsize // 4
        if len(content) > max_content_chars:
            content = content[:max_content_chars - 1]
        if len(metadata_json) > max_metadata_chars:
            metadata_json = metadata_json[:max_metadata_chars - 1]

        self.metadata[idx]["id"] = node.id
        self.metadata[idx]["content"] = content
        self.metadata[idx]["metadata_json"] = metadata_json

        # Flush to disk
        self.embeddings.flush()
        self.metadata.flush()

    def get(self, node_id: int) -> Optional[Node]:
        """Retrieve a node by ID."""
        if node_id not in self._id_to_index:
            return None

        idx = self._id_to_index[node_id]

        # Get embedding from layer 1
        embedding = self.embeddings[idx]["embedding"].copy()

        # Get metadata from layer 2
        content = str(self.metadata[idx]["content"]).strip("\x00") or None
        metadata_json = str(self.metadata[idx]["metadata_json"]).strip("\x00") or None
        metadata = json.loads(metadata_json) if metadata_json else {}

        return Node(
            id=node_id,
            embedding=embedding,
            content=content,
            metadata=metadata,
        )

    def get_embedding(self, node_id: int) -> np.ndarray:
        """Get embedding for a node by ID (returns memmap view for efficiency)."""
        if node_id not in self._id_to_index:
            raise KeyError(f"Node {node_id} not found")
        idx = self._id_to_index[node_id]
        return self.embeddings[idx]["embedding"]

    def get_all_ids(self) -> list[int]:
        """Get all node IDs."""
        return list(self._id_to_index.keys())

    def size(self) -> int:
        """Get the number of nodes."""
        return self._size

    def close(self) -> None:
        """Close storage and release resources."""
        if hasattr(self, "embeddings"):
            del self.embeddings
        if hasattr(self, "metadata"):
            del self.metadata

    def get_next_id(self) -> int:
        """Get the next available node ID."""
        if not self._id_to_index:
            return 0
        return max(self._id_to_index.keys()) + 1

    def add(self, node: Node) -> None:
        """Add a node to storage (alias for save)."""
        self.save(node)

    def delete(self, node_id: int) -> None:
        """Delete a node from storage."""
        if node_id not in self._id_to_index:
            # Node already deleted or doesn't exist
            return
        
        idx = self._id_to_index[node_id]
        
        # Clear the slot
        self.embeddings[idx]["id"] = 0
        self.embeddings[idx]["embedding"] = 0.0
        self.metadata[idx]["id"] = 0
        self.metadata[idx]["content"] = ""
        self.metadata[idx]["metadata_json"] = ""
        
        del self._id_to_index[node_id]
        self._size -= 1
        
        # Flush to disk
        self.embeddings.flush()
        self.metadata.flush()


class DiskNodeStorage(NodeStorage):
    """
    Disk-backed storage using SQLite for metadata and numpy.memmap for embeddings.
    Legacy implementation - consider using MMapNodeStorage instead.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_file: Union[str, Path],
        dim: int,
        capacity: int = 1_000_000,
    ):
        """
        Initialize disk-backed storage.

        Args:
            db_path: Path to SQLite database file
            embedding_file: Path to numpy memmap file for embeddings
            dim: Dimension of embeddings
            capacity: Maximum number of embeddings to store
        """
        self.db_path = Path(db_path)
        self.embedding_file = Path(embedding_file)
        self.dim = dim
        self.capacity = capacity

        # Initialize SQLite database
        self._init_db()

        # Initialize memmap for embeddings
        self._init_memmap()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INTEGER PRIMARY KEY,
                content TEXT,
                metadata TEXT
            )
        """
        )
        self.conn.commit()

    def _init_memmap(self) -> None:
        """Initialize numpy memmap for embeddings."""
        mode = "r+" if self.embedding_file.exists() else "w+"
        self.embeddings = np.memmap(
            self.embedding_file,
            dtype=np.float32,
            mode=mode,
            shape=(self.capacity, self.dim),
        )

        # Track next available index in memmap
        cursor = self.conn.execute("SELECT MAX(node_id) FROM nodes")
        max_id = cursor.fetchone()[0]
        self._next_index = (max_id + 1) if max_id is not None else 0

        self._id_to_index: dict[int, int] = {}
        cursor = self.conn.execute("SELECT node_id FROM nodes ORDER BY node_id")
        for idx, (node_id,) in enumerate(cursor.fetchall()):
            self._id_to_index[node_id] = idx

    def save(self, node: Node) -> None:
        """Save a node to disk storage."""
        # Check capacity
        if self._next_index >= self.capacity:
            raise RuntimeError(f"Storage capacity ({self.capacity}) exceeded")

        # Store embedding in memmap
        memmap_idx = self._next_index
        self.embeddings[memmap_idx] = node.embedding.astype(np.float32)
        self._id_to_index[node.id] = memmap_idx
        self._next_index += 1

        # Store metadata in SQLite
        metadata_json = json.dumps(node.metadata) if node.metadata else None
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes (node_id, content, metadata) VALUES (?, ?, ?)",
            (node.id, node.content, metadata_json),
        )
        self.conn.commit()

    def get(self, node_id: int) -> Optional[Node]:
        """Retrieve a node by ID."""
        cursor = self.conn.execute(
            "SELECT content, metadata FROM nodes WHERE node_id = ?", (node_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        content, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}

        # Get embedding from memmap
        if node_id not in self._id_to_index:
            return None
        memmap_idx = self._id_to_index[node_id]
        embedding = self.embeddings[memmap_idx].copy()

        return Node(
            id=node_id,
            embedding=embedding,
            content=content,
            metadata=metadata,
        )

    def get_embedding(self, node_id: int) -> np.ndarray:
        """Get embedding for a node by ID (returns memmap view for efficiency)."""
        if node_id not in self._id_to_index:
            raise KeyError(f"Node {node_id} not found")
        memmap_idx = self._id_to_index[node_id]
        return self.embeddings[memmap_idx]

    def get_all_ids(self) -> list[int]:
        """Get all node IDs."""
        cursor = self.conn.execute("SELECT node_id FROM nodes ORDER BY node_id")
        return [row[0] for row in cursor.fetchall()]

    def size(self) -> int:
        """Get the number of nodes."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM nodes")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close storage and release resources."""
        if hasattr(self, "conn"):
            self.conn.close()
        if hasattr(self, "embeddings"):
            del self.embeddings  # Close memmap

    def get_next_id(self) -> int:
        """Get the next available node ID."""
        cursor = self.conn.execute("SELECT MAX(node_id) FROM nodes")
        max_id = cursor.fetchone()[0]
        return (max_id + 1) if max_id is not None else 0

    def add(self, node: Node) -> None:
        """Add a node to storage (alias for save)."""
        self.save(node)
