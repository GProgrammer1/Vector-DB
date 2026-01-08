import numpy as np

def top_k_indices_sorted(arr: np.ndarray, k: int):
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]
