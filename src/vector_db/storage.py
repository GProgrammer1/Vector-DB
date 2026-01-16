"""Storage abstraction for vector database nodes."""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict
from pathlib import Path

import numpy as np

from vector_db.types import Node


class NodeStorage(ABC):
    """Abstract base class for node storage backends."""

    @abstractmethod
    def save(self, node: Node) -> None:
        pass

    @abstractmethod
    def get(self, node_id: int) -> Optional[Node]:
        pass

    @abstractmethod
    def get_embedding(self, node_id: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_all_ids(self) -> List[int]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage and release resources."""
        pass

    @abstractmethod
    def add(self, node: Node) -> None:
        pass


class InMemoryNodeStorage(NodeStorage):

    def __init__(self):
        self._nodes: Dict[int, Node] = {}
        self._next_id = 0

    def save(self, node: Node) -> None:
        self._nodes[node.id] = node
        if node.id >= self._next_id:
            self._next_id = node.id + 1

    def add(self, node: Node) -> None:
        self.save(node)

    def get(self, node_id: int) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_embedding(self, node_id: int) -> np.ndarray:
        node = self.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} not found")
        return node.embedding

    def get_all_ids(self) -> List[int]:
        return list(self._nodes.keys())

    def size(self) -> int:
        return len(self._nodes)

    def close(self) -> None:
        pass

    def get_next_id(self) -> int:
        return self._next_id


class MMapNodeStorage(NodeStorage):
    """
    Two-layer memmap storage for scalable vector database.
    
    Layer 1: Embeddings + IDs (memmap)
    Layer 2: IDs + Content + Metadata (memmap with structured dtype)
    
    """

    def __init__(
        self,
        embedding_file: Union[str, Path],
        metadata_file: Union[str, Path],
        dim: int,
        capacity: int = 1_000_000,
    ):
        """
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

        self._init_embedding_memmap()
        self._init_metadata_memmap()

    def _init_embedding_memmap(self) -> None:
        mode = "r+" if self.embedding_file.exists() else "w+"
        dtype: np.dtype = np.dtype([("id", np.int64), ("embedding", np.float32, (self.dim,))])
        self.embeddings: np.memmap = np.memmap(
            self.embedding_file,
            dtype=dtype,
            mode=mode,
            shape=(self.capacity,),
        )

        self._id_to_index: Dict[int, int] = {}
        if mode == "r+":
            valid_indices = []
            for idx in range(self.capacity):
                node_id = int(self.embeddings[idx]["id"])
                embedding = self.embeddings[idx]["embedding"]
                if node_id >= 0 and np.any(embedding != 0.0):
                    valid_indices.append(idx)
                    self._id_to_index[node_id] = idx
            self._size = len(valid_indices)
        else:
            self._size = 0

    def _init_metadata_memmap(self) -> None:
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
        if self._size >= self.capacity:
            raise RuntimeError(f"Storage capacity ({self.capacity}) exceeded")
        
        for idx in range(self.capacity):
            node_id = int(self.embeddings[idx]["id"])
            embedding = self.embeddings[idx]["embedding"]
            if node_id < 0 or np.all(embedding == 0.0):
                return idx
        
        return self._size

    def save(self, node: Node) -> None:
        if node.id in self._id_to_index:
            idx = self._id_to_index[node.id]
        else:
            idx = self._find_free_slot()
            self._id_to_index[node.id] = idx
            self._size += 1

        self.embeddings[idx]["id"] = node.id
        self.embeddings[idx]["embedding"] = node.embedding.astype(np.float32)

        metadata_json = json.dumps(node.metadata) if node.metadata else ""
        content = node.content or ""
        
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
        if node_id not in self._id_to_index:
            raise KeyError(f"Node {node_id} not found")
        idx = self._id_to_index[node_id]
        return self.embeddings[idx]["embedding"]

    def get_all_ids(self) -> List[int]:
        return list(self._id_to_index.keys())

    def size(self) -> int:
        return self._size

    def close(self) -> None:
        if hasattr(self, "embeddings"):
            del self.embeddings
        if hasattr(self, "metadata"):
            del self.metadata

    def get_next_id(self) -> int:
        if not self._id_to_index:
            return 0
        return max(self._id_to_index.keys()) + 1

    def add(self, node: Node) -> None:
        self.save(node)

    def delete(self, node_id: int) -> None:
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


# class DiskNodeStorage(NodeStorage):
#     """
#     Disk-backed storage using SQLite for metadata and numpy.memmap for embeddings.
#     Legacy implementation - consider using MMapNodeStorage instead.
#     """

#     def __init__(
#         self,
#         db_path: Union[str, Path],
#         embedding_file: Union[str, Path],
#         dim: int,
#         capacity: int = 1_000_000,
#     ):
#         """

#         Args:
#             db_path: Path to SQLite database file
#             embedding_file: Path to numpy memmap file for embeddings
#             dim: Dimension of embeddings
#             capacity: Maximum number of embeddings to store
#         """
#         self.db_path = Path(db_path)
#         self.embedding_file = Path(embedding_file)
#         self.dim = dim
#         self.capacity = capacity

#         # Initialize SQLite database
#         self._init_db()

#         self._init_memmap()

#     def _init_db(self) -> None:
#         self.conn = sqlite3.connect(self.db_path)
#         self.conn.execute(
#             """
#             CREATE TABLE IF NOT EXISTS nodes (
#                 node_id INTEGER PRIMARY KEY,
#                 content TEXT,
#                 metadata TEXT
#             )
#         """
#         )
#         self.conn.commit()

#     def _init_memmap(self) -> None:
#         mode = "r+" if self.embedding_file.exists() else "w+"
#         self.embeddings = np.memmap(
#             self.embedding_file,
#             dtype=np.float32,
#             mode=mode,
#             shape=(self.capacity, self.dim),
#         )

#         # Track next available index in memmap
#         cursor = self.conn.execute("SELECT MAX(node_id) FROM nodes")
#         max_id = cursor.fetchone()[0]
#         self._next_index = (max_id + 1) if max_id is not None else 0

#         self._id_to_index: Dict[int, int] = {}
#         cursor = self.conn.execute("SELECT node_id FROM nodes ORDER BY node_id")
#         for idx, (node_id,) in enumerate(cursor.fetchall()):
#             self._id_to_index[node_id] = idx

#     def save(self, node: Node) -> None:
#         if self._next_index >= self.capacity:
#             raise RuntimeError(f"Storage capacity ({self.capacity}) exceeded")

#         # Store embedding in memmap
#         memmap_idx = self._next_index
#         self.embeddings[memmap_idx] = node.embedding.astype(np.float32)
#         self._id_to_index[node.id] = memmap_idx
#         self._next_index += 1

#         # Store metadata in SQLite
#         metadata_json = json.dumps(node.metadata) if node.metadata else None
#         self.conn.execute(
#             "INSERT OR REPLACE INTO nodes (node_id, content, metadata) VALUES (?, ?, ?)",
#             (node.id, node.content, metadata_json),
#         )
#         self.conn.commit()

#     def get(self, node_id: int) -> Optional[Node]:
#         cursor = self.conn.execute(
#             "SELECT content, metadata FROM nodes WHERE node_id = ?", (node_id,)
#         )
#         row = cursor.fetchone()
#         if row is None:
#             return None

#         content, metadata_json = row
#         metadata = json.loads(metadata_json) if metadata_json else {}

#         # Get embedding from memmap
#         if node_id not in self._id_to_index:
#             return None
#         memmap_idx = self._id_to_index[node_id]
#         embedding = self.embeddings[memmap_idx].copy()

#         return Node(
#             id=node_id,
#             embedding=embedding,
#             content=content,
#             metadata=metadata,
#         )

#     def get_embedding(self, node_id: int) -> np.ndarray:
#         if node_id not in self._id_to_index:
#             raise KeyError(f"Node {node_id} not found")
#         memmap_idx = self._id_to_index[node_id]
#         return self.embeddings[memmap_idx]

#     def get_all_ids(self) -> List[int]:
#         cursor = self.conn.execute("SELECT node_id FROM nodes ORDER BY node_id")
#         return [row[0] for row in cursor.fetchall()]

#     def size(self) -> int:
#         cursor = self.conn.execute("SELECT COUNT(*) FROM nodes")
#         return cursor.fetchone()[0]

#     def close(self) -> None:
#         if hasattr(self, "conn"):
#             self.conn.close()
#         if hasattr(self, "embeddings"):
#             del self.embeddings  # Close memmap

#     def get_next_id(self) -> int:
#         cursor = self.conn.execute("SELECT MAX(node_id) FROM nodes")
#         max_id = cursor.fetchone()[0]
#         return (max_id + 1) if max_id is not None else 0

#     def add(self, node: Node) -> None:
#         self.save(node)
