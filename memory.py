# memory.py - episodic memory for topos
#
# stores text + metadata in an in-memory chromadb collection
# each entry gets a monotonically increasing integer id
#
# usage:
#   EpisodicMemory(embedder)
#     .write(text, metadata)  -> None
#     .recall(query, n=3)     -> list of dicts with "text", "similarity", "turn"

import uuid

import chromadb
import config
from embedder import Embedder


class EpisodicMemory:
    # episodic memory backed by in-memory chromadb

    def __init__(self, embedder):
        self._embedder = embedder
        self._client = chromadb.Client()          # in-memory, no persistence
        # unique name per instance because chromadb shares global state
        # so two workspaces would stomp on each other without this
        col_name = f"{config.MEMORY_COLLECTION}_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},   # use cosine distance
        )
        self._next_id = 0

    def write(self, text, metadata):
        # embed the text and store it with metadata
        # metadata needs to have a "turn" key or it blows up
        if "turn" not in metadata:
            raise ValueError("metadata must include a 'turn' key (int).")

        vec = self._embedder.embed(text).tolist()
        doc_id = str(self._next_id)
        self._next_id += 1

        self._collection.add(
            ids=[doc_id],
            embeddings=[vec],
            documents=[text],
            metadatas=[metadata],
        )

    def recall(self, query, n=3):
        # returns the top n memories most similar to query
        # each result is a dict: text, similarity, turn
        count = self._collection.count()
        if count == 0:
            return []

        k = min(n, count)
        vec = self._embedder.embed(query).tolist()

        results = self._collection.query(
            query_embeddings=[vec],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        memories = []
        for doc, distance, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # chromadb cosine space gives distance = 1 - cosine_similarity
            similarity = 1.0 - float(distance)
            memories.append(
                {
                    "text": doc,
                    "similarity": round(similarity, 4),
                    "turn": meta.get("turn"),
                }
            )

        return memories


# demo
if __name__ == "__main__":
    emb = Embedder()
    mem = EpisodicMemory(emb)

    sentences = [
        ("The dog chased the ball across the park.", {"turn": 1, "source": "demo"}),
        ("Quantum computing promises exponential speed-ups for certain problems.", {"turn": 2, "source": "demo"}),
        ("She enjoyed a warm cup of coffee on a rainy afternoon.", {"turn": 3, "source": "demo"}),
        ("Neural networks learn representations from raw data.", {"turn": 4, "source": "demo"}),
        ("The puppy fetched the frisbee and wagged its tail.", {"turn": 5, "source": "demo"}),
    ]

    print("=== Writing memories ===")
    for text, meta in sentences:
        mem.write(text, meta)
        print(f"  [turn {meta['turn']}] {text}")

    query = "a dog playing fetch outdoors"
    print(f"\n=== Recalling top 2 for: '{query}' ===")
    results = mem.recall(query, n=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [turn={r['turn']} | sim={r['similarity']:.4f}] {r['text']}")
