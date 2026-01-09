import numpy as np

def euclidean_vector_distance(v1: np.ndarray, v2: np.ndarray):
   return np.linalg.norm(v1 - v2)