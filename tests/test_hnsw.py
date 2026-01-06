import numpy as np
import pytest
from vector_db.indexing.hnsw import HNSW, euclidean_vector_distance

def test_initialization():
    rng = np.random.RandomState(42)
    hnsw = HNSW(M=16, ef_construction=100, rng=rng)
    assert hnsw.M == 16
    assert hnsw.node_store == {}
    assert hnsw.max_level == -1
    assert hnsw.entry_node_id is None

def test_insert_single_node():
    rng = np.random.RandomState(42)
    hnsw = HNSW(M=16, ef_construction=100, rng=rng)
    vec = np.array([1.0, 2.0], dtype=np.float32)
    hnsw.insert_node(0, vec)
    
    assert 0 in hnsw.node_store
    assert hnsw.entry_node_id == 0
    assert hnsw.max_level >= 0
    assert np.array_equal(hnsw.node_store[0].embedding, vec)

def brute_force_search(query, data, k):
    distances = []
    for idx, vec in data.items():
        d = np.linalg.norm(query - vec)
        distances.append((d, idx))
    distances.sort()
    return distances[:k]

def test_recall_small_dataset():
    dim = 8
    num_elements = 200
    M = 16
    ef_construction = 50
    rng = np.random.RandomState(42)
    
    hnsw = HNSW(M=M, ef_construction=ef_construction, rng=rng)
    data = {}
    
    # Insert random vectors
    for i in range(num_elements):
        vec = np.random.rand(dim).astype(np.float32)
        data[i] = vec
        hnsw.insert_node(i, vec)
        
    assert len(hnsw.node_store) == num_elements
    
    # Run queries
    num_queries = 10
    k = 5
    ef_search = 50
    
    passes = 0
    for _ in range(num_queries):
        query = np.random.rand(dim).astype(np.float32)
        
        gt_results = brute_force_search(query, data, k)
        gt_ids = {idx for _, idx in gt_results}
        
        # HNSW search
        curr_id = hnsw.entry_node_id
        for level in range(hnsw.max_level, 0, -1):
            curr_id = hnsw._greedy_search_level(query, curr_id, level)
        candidates = hnsw._search_layer(query, curr_id, ef_search, 0)
        hnsw_ids = {idx for _, idx in candidates[:k]}
        
        recall = len(gt_ids.intersection(hnsw_ids)) / k
        if recall >= 0.8: 
            passes += 1
            
    # We expect high recall
    assert passes >= num_queries * 0.9

def test_connectivity():
    # Ensure graph is not disconnected (all nodes reachable from entry point at least at level 0)
    # BFS from entry node at level 0
    dim = 4
    num_elements = 50
    rng = np.random.RandomState(42)
    hnsw = HNSW(M=8, ef_construction=50, rng=rng)
    
    for i in range(num_elements):
        vec = np.random.rand(dim).astype(np.float32)
        hnsw.insert_node(i, vec)
        
    visited = set()
    queue = [hnsw.entry_node_id]
    visited.add(hnsw.entry_node_id)
    
    while queue:
        curr = queue.pop(0)
        node = hnsw.node_store[curr]
        for neighbor in node.neighbors.get(0, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    # If not fully connected, check if > 90% connected.
    assert len(visited) > num_elements * 0.9
