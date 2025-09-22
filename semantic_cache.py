import time
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """
    Semantic cache system using embeddings for approximate matching.
    Stores queries and responses with TTL.
    """

    def __init__(self, cache_file="semantic_cache.json", threshold=0.50, default_ttl=3600):
        self.cache_file = cache_file
        self.threshold = threshold
        self.default_ttl = default_ttl
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    return [
                        {
                            "embedding": np.array(item["embedding"]),
                            "response": item["response"],
                            "timestamp": item["timestamp"],
                            "ttl": item["ttl"],
                        }
                        for item in raw
                    ]
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save_cache(self):
        raw = [
            {
                "embedding": emb["embedding"].tolist(),
                "response": emb["response"],
                "timestamp": emb["timestamp"],
                "ttl": emb["ttl"],
            }
            for emb in self.cache
        ]
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(raw, f)

    def _cleanup(self):
        now = time.time()
        self.cache = [item for item in self.cache if now - item["timestamp"] <= item["ttl"]]

    def get(self, query: str, threshold: float = None):
        """Retrieve response from cache if query is semantically similar."""
        self._cleanup()

        if not self.cache:
            return None

        query_emb = self.model.encode([query])
        embeddings = np.array([item["embedding"] for item in self.cache])
        sims = cosine_similarity(query_emb, embeddings)[0]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        threshold = threshold if threshold is not None else self.threshold

        if best_score >= threshold:
            return self.cache[best_idx]["response"]
        return None

    def set(self, query: str, response: str, ttl=None):
        """Store query-response pair in cache."""
        query_emb = self.model.encode([query])[0]
        ttl = ttl if ttl is not None else self.default_ttl
        self.cache.append({
            "embedding": query_emb,
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl
        })
        self._save_cache()
