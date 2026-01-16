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
        
        Args:
            storage: MMapNodeStorage instance for node data
            config_path: Path to config.yaml for index parameters
            index_file: Path to save/load HNSW index state (optional)
        """
        self.storage = storage
        self.config_path = Path(config_path)
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if "index" not in config:
            raise ValueError("Config file must contain 'index' section")
        index_config = config["index"]
        
        if "M" not in index_config:
            raise ValueError("Config 'index.M' is required")
        M = index_config["M"]
        
        if "ef_construction" not in index_config:
            raise ValueError("Config 'index.ef_construction' is required")
        ef_construction = index_config["ef_construction"]
        
        if index_file:
            self.index_file = Path(index_file)
        else:
            base_path = Path(storage.embedding_file).parent / Path(storage.embedding_file).stem.replace(".embeddings", "")
            self.index_file = base_path.with_suffix(".index.pkl")
        rng = random.Random(42)
        self.index = HNSW(
            M=M,
            ef_construction=ef_construction,
            rng=rng,
            storage=storage,
            index_file=self.index_file,
        )
        
        self._index_loaded = self.index_file.exists()
        self._index_modified = False
        
        if "flush_threshold" not in index_config:
            raise ValueError("Config 'index.flush_threshold' is required")
        self.flush_threshold = index_config["flush_threshold"]

    def is_index_loaded(self) -> bool:
        return self._index_loaded

    def insert_node(self, node: Node) -> None:
        self.index.insert_node(node)
        self._index_modified = True
        
        if self._should_flush():
            self.save_index()

    def delete_node(self, node_id: int) -> None:
        self.index.delete_node(node_id)
        self._index_modified = True

    def search(
        self, query: np.ndarray, k: int, **kwargs
    ) -> List[Tuple[Node, float]]:
        return self.index.search(query, k=k, **kwargs)

    def save_index(self) -> None:
        if self._index_modified:
            self.index.save_index()
            self._index_modified = False

    def force_save_index(self) -> None:
        self.index.save_index()
        self._index_modified = False

    def get_index_size(self) -> int:
        return len(self.index.node_store)

    def _should_flush(self) -> bool:
        return self.get_index_size() >= self.flush_threshold

