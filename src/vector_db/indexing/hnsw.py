import numpy as np
import random
import math
import heapq
from typing import Optional, List, Set, Tuple

# Assuming euclidean_vector_distance is imported or defined
def euclidean_vector_distance(a, b):
    return np.linalg.norm(a - b)

class HNSW:
    class Node:
        def __init__(self, id: int, embedding: np.ndarray):
            self.id = id
            self.embedding = embedding
            # dict: level -> list of neighbor IDs
            self.neighbors: dict[int, list[int]] = {}

    def __init__(self, M: int, ef_construction: int, rng: random.Random):
        self.M = M
        self.M_max = M  
        self.M_max0 = M * 2  
        self.ef_construction = ef_construction
        self.rng = rng
        self.node_store: dict[int, HNSW.Node] = {}
        self.entry_node_id: Optional[int] = None
        self.max_level = -1
        # Level multiplier
        self.level_mult = 1 / math.log(M)

    def sample_level(self) -> int:
        return int(-math.log(self.rng.random()) * self.level_mult)

    def _search_layer(self, query_embedding: np.ndarray, entry_id: int, ef: int, level: int) -> List[Tuple[float, int]]:
        """
        The core Best-First Search algorithm using a Priority Queue.
        Returns the 'ef' closest nodes found in the given level.
        """
        # Min-heap of candidates to explore (distance, id)
        candidates = [(euclidean_vector_distance(query_embedding, self.node_store[entry_id].embedding), entry_id)]
        # Max-heap of best nodes found so far ( -distance, id) 
        found_nodes = [(-candidates[0][0], entry_id)]
        
        visited = {entry_id}
        heapq.heapify(candidates)
        

        while candidates:
            dist, curr_id = heapq.heappop(candidates)
            
            # If this candidate is further than the worst in our result list, stop
            if dist > -found_nodes[0][0]:
                break

            curr_node = self.node_store[curr_id]
            for neighbor_id in curr_node.neighbors.get(level, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    n_dist = euclidean_vector_distance(query_embedding, self.node_store[neighbor_id].embedding)
                    
                    # If neighbor is closer than the worst found or we haven't reached ef capacity
                    if n_dist < -found_nodes[0][0] or len(found_nodes) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor_id))
                        heapq.heappush(found_nodes, (-n_dist, neighbor_id))
                        
                        if len(found_nodes) > ef:
                            heapq.heappop(found_nodes)

        # Return sorted list of (distance, id)
        return [( -d, idx) for d, idx in sorted(found_nodes, reverse=True)]

    def _select_neighbors(self, candidates_with_dist: List[Tuple[float, int]], k: int) -> List[int]:
        """Simple selection: returns k closest IDs from candidates_with_dist."""
        # Candidates are already (dist, id)
        candidates_with_dist.sort()
        return [node_id for _, node_id in candidates_with_dist[:k]]

    def insert_node(self, node_id: int, embedding: np.ndarray):
        new_node = self.Node(node_id, embedding)
        self.node_store[node_id] = new_node
        
        target_level = self.sample_level()
        curr_max_level = self.max_level
        
        # Handle empty graph
        if self.entry_node_id is None:
            self.entry_node_id = node_id
            self.max_level = target_level
            for l in range(target_level + 1):
                new_node.neighbors[l] = []
            return

        #  Greedy descent to find entry point at target_level
        curr_id = self.entry_node_id
        for level in range(curr_max_level, target_level, -1):
            curr_id = self._greedy_search_level(embedding, curr_id, level)

        # best-first search
        
        start_level = min(target_level, curr_max_level)
        for level in range(start_level, -1, -1):
            # Find the best candidates using Best-first search
            candidates = self._search_layer(embedding, curr_id, self.ef_construction, level)
            
            # Select M neighbors to connect to
            m_limit = self.M_max0 if level == 0 else self.M_max
            selected_neighbors = self._select_neighbors(candidates, self.M)
            
            # Add bidirectional connections
            new_node.neighbors[level] = selected_neighbors
            for neighbor_id in selected_neighbors:
                neighbor_node = self.node_store[neighbor_id]
                if level not in neighbor_node.neighbors: neighbor_node.neighbors[level] = []
                neighbor_node.neighbors[level].append(node_id)
                
                # Prune neighbor connections if they exceed limit
                if len(neighbor_node.neighbors[level]) > m_limit:
                    all_n = []
                    for nid in neighbor_node.neighbors[level]:
                        d = euclidean_vector_distance(neighbor_node.embedding, self.node_store[nid].embedding)
                        all_n.append((d, nid))
                    neighbor_node.neighbors[level] = self._select_neighbors(all_n, m_limit)

            # Closest candidate is the entry point for the next level down
            curr_id = selected_neighbors[0]

        #  Update global entry point if new node is higher than current max
        if target_level > self.max_level:
            self.max_level = target_level
            self.entry_node_id = node_id

    def _greedy_search_level(self, query_embedding: np.ndarray, curr_id: int, level: int) -> int:
        """Simple 1-best greedy search for upper layers."""
        curr_dist = euclidean_vector_distance(query_embedding, self.node_store[curr_id].embedding)
        while True:
            best_neighbor_id = curr_id
            for neighbor_id in self.node_store[curr_id].neighbors.get(level, []):
                d = euclidean_vector_distance(query_embedding, self.node_store[neighbor_id].embedding)
                if d < curr_dist:
                    curr_dist = d
                    best_neighbor_id = neighbor_id
            
            if best_neighbor_id == curr_id:
                return curr_id
            curr_id = best_neighbor_id