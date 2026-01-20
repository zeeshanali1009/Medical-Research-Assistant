import os
import numpy as np
import faiss
import pickle
from typing import List, Dict


class FaissStore:
    def __init__(self, dim: int, index_path: str = None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # dot-product similarity (use normalized vectors)
        self.metadata = []
        self.index_path = index_path

    def add(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        arr = np.vstack(vectors).astype('float32')
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        q = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta.update({"score": float(score)})
            results.append(meta)
        return results

    def save(self, path: str):
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".meta", "wb") as fh:
            pickle.dump(self.metadata, fh)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".meta", "rb") as fh:
            self.metadata = pickle.load(fh)
