# embedder.py - sentence embedding and cosine similarity for topos
#
# wraps a sentence-transformers model (loaded once) and gives you:
#   embed(text)       -> normalised 1d vector (shape = WORKSPACE_DIM,)
#   similarity(a, b)  -> cosine similarity (float between -1 and 1)

import logging
import numpy as np

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

import config


class Embedder:
    # thin wrapper around sentencetransformer

    def __init__(self):
        self.model = SentenceTransformer(config.EMBED_MODEL)

    def embed(self, text):
        # returns a normalised 1d embedding of the text
        # shape is always (config.WORKSPACE_DIM,)
        vec = self.model.encode(text, convert_to_numpy=True)
        vec = vec.flatten()[:config.WORKSPACE_DIM]          # safety clip
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def similarity(self, a, b):
        # cosine similarity between two vectors
        dot = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return dot / denom


# quick test
if __name__ == "__main__":
    emb = Embedder()

    s1 = "The cat sat on the mat."
    s2 = "A kitten rested on the rug."

    v1 = emb.embed(s1)
    v2 = emb.embed(s2)

    print(f"s1          : {s1}")
    print(f"s2          : {s2}")
    print(f"v1 shape    : {v1.shape}")
    print(f"v2 shape    : {v2.shape}")
    print(f"v1 norm     : {np.linalg.norm(v1):.6f}")
    print(f"v2 norm     : {np.linalg.norm(v2):.6f}")
    print(f"similarity  : {emb.similarity(v1, v2):.4f}")
