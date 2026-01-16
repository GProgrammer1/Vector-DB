import json
import pickle
import numpy as np
import random
import math
import heapq
from pathlib import Path
from typing import Optional, List, Set, Tuple, Union, Dict

from ..types import Node
from ..util.distance import euclidean_vector_distance
from ..storage import NodeStorage, InMemoryNodeStorage


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) index.
    
    The index only stores node IDs and graph structure (neighbors).
    Embeddings and metadata are stored separately in NodeStorage.
    """

    class InternalNode:
        """Internal representation: only stores node ID and neighbors (graph structure)."""

        def __init__(self, node_id: int):
            self.id = node_id
            self.neighbors: Dict[int, List[int]] = {} 

    def __init__(
        self,
        M: int,
        ef_construction: int,
        rng: random.Random,
        storage: Optional[NodeStorage] = None,
        index_file: Optional[Union[str, Path]] = None,
    ):
        """

        Args:
            M: Maximum number of connections per node
            ef_construction: Number of candidates to consider during construction
            rng: Random number generator
            storage: NodeStorage backend for embeddings and metadata
            index_file: Optional path to save/load index state
        """
        self.M = M
        self.M_max = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.rng = rng
        self.storage = storage or InMemoryNodeStorage()
        self.index_file = Path(index_file) if index_file else None

        self.node_store: Dict[int, HNSW.InternalNode] = {} 
        self.entry_node_id: Optional[int] = None
        self.max_level = -1
        self.level_mult = 1 / math.log(M)

        if self.index_file and self.index_file.exists():
            self.load_index()

    def sample_level(self) -> int:
        return int(-math.log(self.rng.random()) * self.level_mult)

    def _get_embedding(self, node_id: int) -> np.ndarray:
        try:
            return self.storage.get_embedding(node_id)
        except KeyError:
            raise ValueError(f"Node {node_id} not found in storage (may have been deleted)")

    def _search_layer(
        self, query_embedding: np.ndarray, entry_id: int, ef: int, level: int,
        filter_ids: Optional[Set[int]] = None
    ) -> List[Tuple[float, int]]:
        """
        The core Best-First Search algorithm using a Priority Queue.
        Returns the 'ef' closest nodes found in the given level.
        If filter_ids is provided, only nodes in that set are returned as results,
        but non-matching nodes are still used for navigation.
        """
        # Min-heap of candidates to explore (distance, id)
        dist = euclidean_vector_distance(query_embedding, self._get_embedding(entry_id))
        candidates = [(dist, entry_id)]
        found_nodes = []
        if filter_ids is None or entry_id in filter_ids:
            found_nodes.append((-dist, entry_id))

        visited = {entry_id}
        heapq.heapify(candidates)

        while candidates:
            dist, curr_id = heapq.heappop(candidates)

            if len(found_nodes) >= ef and dist > -found_nodes[0][0]:
                break

            curr_node = self.node_store[curr_id]
            for neighbor_id in curr_node.neighbors.get(level, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    try:
                        n_dist = euclidean_vector_distance(
                            query_embedding, self._get_embedding(neighbor_id)
                        )
                    except (KeyError, ValueError):
                        continue

                    heapq.heappush(candidates, (n_dist, neighbor_id))
                    
                    if filter_ids is None or neighbor_id in filter_ids:
                        if len(found_nodes) < ef or n_dist < -found_nodes[0][0]:
                            heapq.heappush(found_nodes, (-n_dist, neighbor_id))
                            if len(found_nodes) > ef:
                                heapq.heappop(found_nodes)

        # Return sorted list of (distance, id)
        return [(-d, idx) for d, idx in sorted(found_nodes, reverse=True)]

    def _select_neighbors(
        self, candidates_with_dist: List[Tuple[float, int]], k: int
    ) -> List[int]:
        """Returns k closest IDs from candidates_with_dist."""
        candidates_with_dist.sort()
        return [node_id for _, node_id in candidates_with_dist[:k]]

    def insert_node(self, node: Node) -> None:
        """        
        The node is first saved to storage, then added to the graph structure.
        """
        self.storage.save(node)

        if node.id in self.node_store:
            return

        new_node = self.InternalNode(node.id)
        self.node_store[node.id] = new_node

        target_level = self.sample_level()
        curr_max_level = self.max_level

        if self.entry_node_id is None:
            self.entry_node_id = node.id
            self.max_level = target_level
            for l in range(target_level + 1):
                new_node.neighbors[l] = []
            if self.index_file:
                self.save_index()
            return

        curr_id = self.entry_node_id
        for level in range(curr_max_level, target_level, -1):
            curr_id = self._greedy_search_level(node.embedding, curr_id, level)

        start_level = min(target_level, curr_max_level)
        for level in range(start_level, -1, -1):
            candidates = self._search_layer(
                node.embedding, curr_id, self.ef_construction, level
            )

            m_limit = self.M_max0 if level == 0 else self.M_max
            selected_neighbors = self._select_neighbors(candidates, self.M)

            new_node.neighbors[level] = selected_neighbors
            for neighbor_id in selected_neighbors:
                neighbor_node = self.node_store[neighbor_id]
                if level not in neighbor_node.neighbors:
                    neighbor_node.neighbors[level] = []
                neighbor_node.neighbors[level].append(node.id)

                if len(neighbor_node.neighbors[level]) > m_limit:
                    all_n = []
                    for nid in neighbor_node.neighbors[level]:
                        try:
                            d = euclidean_vector_distance(
                                self._get_embedding(neighbor_node.id),
                                self._get_embedding(nid),
                            )
                            all_n.append((d, nid))
                        except (KeyError, ValueError):
                            continue
                    neighbor_node.neighbors[level] = self._select_neighbors(
                        all_n, m_limit
                    )

            curr_id = selected_neighbors[0]

        if target_level > self.max_level:
            self.max_level = target_level
            self.entry_node_id = node.id

        if self.index_file:
            self.save_index()

    def delete_node(self, node_id: int) -> None:
        """
        Delete a node from the index.
        
        Removes the node from graph structure and storage.
        """
        if node_id not in self.node_store:
            return

        # Remove from graph: remove all connections
        node = self.node_store[node_id]
        for level, neighbors in node.neighbors.items():
            for neighbor_id in neighbors:
                if neighbor_id in self.node_store:
                    neighbor_node = self.node_store[neighbor_id]
                    if level in neighbor_node.neighbors:
                        neighbor_node.neighbors[level] = [
                            nid for nid in neighbor_node.neighbors[level] if nid != node_id
                        ]

        del self.node_store[node_id]

        if self.entry_node_id == node_id:
            if self.node_store:
                self.entry_node_id = max(
                    self.node_store.keys(),
                    key=lambda nid: max(self.node_store[nid].neighbors.keys())
                    if self.node_store[nid].neighbors
                    else -1,
                )
                self.max_level = max(
                    self.node_store[self.entry_node_id].neighbors.keys()
                ) if self.node_store[self.entry_node_id].neighbors else -1
            else:
                self.entry_node_id = None
                self.max_level = -1

        if hasattr(self.storage, "delete"):
            self.storage.delete(node_id)

        if self.index_file:
            self.save_index()

    def build_index(self, nodes: List[Node]) -> None:
       
        for node in nodes:
            self.insert_node(node)

    def _greedy_search_level(
        self, query_embedding: np.ndarray, curr_id: int, level: int
    ) -> int:
        try:
            curr_dist = euclidean_vector_distance(
                query_embedding, self._get_embedding(curr_id)
            )
        except (KeyError, ValueError):
            if not self.node_store:
                return curr_id
            for nid in self.node_store.keys():
                try:
                    if level in self.node_store[nid].neighbors:
                        return nid
                except KeyError:
                    continue
            return curr_id
        
        while True:
            best_neighbor_id = curr_id
            if curr_id not in self.node_store:
                return curr_id
            for neighbor_id in self.node_store[curr_id].neighbors.get(level, []):
                try:
                    d = euclidean_vector_distance(
                        query_embedding, self._get_embedding(neighbor_id)
                    )
                    if d < curr_dist:
                        curr_dist = d
                        best_neighbor_id = neighbor_id
                except (KeyError, ValueError):
                    continue

            if best_neighbor_id == curr_id:
                return curr_id
            curr_id = best_neighbor_id

    def search(
        self, query: np.ndarray, k: int, **kwargs
    ) -> List[Tuple[Node, float]]:
        ef = kwargs.get("ef", 50)
        filter_ids = kwargs.get("filter_ids")
        if self.entry_node_id is None:
            return []

        try:
            self._get_embedding(self.entry_node_id)
        except (KeyError, ValueError):
            valid_nodes = [nid for nid in self.node_store.keys() 
                          if self.storage.get(nid) is not None]
            if not valid_nodes:
                return []
            self.entry_node_id = valid_nodes[0]

        curr_id = self.entry_node_id
        for level in range(self.max_level, 0, -1):
            curr_id = self._greedy_search_level(query, curr_id, level)

        try:
            candidates = self._search_layer(query, curr_id, ef, 0, filter_ids=filter_ids)
        except (KeyError, ValueError):
            return []

        candidates.sort()

        results = []
        for dist, node_id in candidates[:k]:
            node = self.storage.get(node_id)
            if node:
                results.append((node, dist))

        return results

    def save_index(self) -> None:
        if self.index_file is None:
            return

        index_data = {
            "M": self.M,
            "M_max": self.M_max,
            "M_max0": self.M_max0,
            "ef_construction": self.ef_construction,
            "max_level": self.max_level,
            "entry_node_id": self.entry_node_id,
            "node_store": {
                node_id: {
                    "id": node.id,
                    "neighbors": node.neighbors,
                }
                for node_id, node in self.node_store.items()
            },
        }

        with open(self.index_file, "wb") as f:
            pickle.dump(index_data, f)

    def load_index(self) -> None:
        if self.index_file is None or not self.index_file.exists():
            return

        with open(self.index_file, "rb") as f:
            index_data = pickle.load(f)

        self.M = index_data["M"]
        self.M_max = index_data["M_max"]
        self.M_max0 = index_data["M_max0"]
        self.ef_construction = index_data["ef_construction"]
        self.max_level = index_data["max_level"]
        self.entry_node_id = index_data["entry_node_id"]

        # Reconstruct node_store
        self.node_store = {}
        for node_id, node_data in index_data["node_store"].items():
            internal_node = self.InternalNode(node_data["id"])
            internal_node.neighbors = node_data["neighbors"]
            self.node_store[node_id] = internal_node

        # Recalculate level_mult
        self.level_mult = 1 / math.log(self.M)
