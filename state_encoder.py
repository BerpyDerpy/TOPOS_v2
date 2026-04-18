# state_encoder.py - state encoder for topos
#
# produces a fixed size latent vector z_t in R^448 from the four
# workspace submodules: concept graph, affective state, episodic memory,
# and the current input embedding
#
# z_t = [z_graph | z_affect | z_memory | z_input]
#        R^128     R^64       R^128      R^128
#
# the only trainable params are four linear projections (no bias, no
# activation). they're initialised orthogonally so outputs start at
# roughly unit variance when inputs are unit normalised.
#
# designed to be trained later via a forward model loss (predict z_{t+1}
# from z_t), so smoothness and consistency across turns matter most.

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config
from embedder import Embedder
from graph import ConceptGraph
from affect import AffectiveState
from memory import EpisodicMemory


# dimension stuff
# source dimensions (from existing modules)
_SRC_EMBED = config.WORKSPACE_DIM    # 384 from minilm
_SRC_AFFECT = config.AFFECT_DIM      # 64 affect state vector

# target subvector dimensions
_TGT_GRAPH  = 128
_TGT_AFFECT = 64
_TGT_MEMORY = 128
_TGT_INPUT  = 128

Z_DIM = _TGT_GRAPH + _TGT_AFFECT + _TGT_MEMORY + _TGT_INPUT   # 448

# defaults
_DEFAULT_TOP_K = 10            # concepts to grab from graph
_RECENCY_HALFLIFE = 10.0       # turns for exponential decay half-life


class StateEncoder(nn.Module):
    # encodes the topos workspace state into a fixed size latent vector z_t
    #
    # embedder: shared sentence transformer wrapper
    # top_k: max number of graph nodes to aggregate for z_graph
    # recency_halflife: half-life in turns for memory recency weighting
    #   10 means a memory 10 turns old gets half the recency weight

    def __init__(self, embedder, top_k=_DEFAULT_TOP_K, recency_halflife=_RECENCY_HALFLIFE):
        super().__init__()
        self._embedder = embedder
        self._top_k = top_k
        self._recency_decay = np.log(2.0) / recency_halflife

        # learned projections (no bias, no activation)
        self.proj_graph  = nn.Linear(_SRC_EMBED,  _TGT_GRAPH,  bias=False)
        self.proj_affect = nn.Linear(_SRC_AFFECT,  _TGT_AFFECT, bias=False)
        self.proj_memory = nn.Linear(_SRC_EMBED,  _TGT_MEMORY, bias=False)
        self.proj_input  = nn.Linear(_SRC_EMBED,  _TGT_INPUT,  bias=False)

        # orthogonal init preserves input norms through the projection
        for proj in (self.proj_graph, self.proj_affect,
                     self.proj_memory, self.proj_input):
            nn.init.orthogonal_(proj.weight)

    @torch.no_grad()
    def encode(self, graph, affect, memory, input_embedding, current_turn=None):
        # produce the latent state vector z_t in R^448
        #
        # graph: the ConceptGraph
        # affect: the AffectiveState
        # memory: the EpisodicMemory
        # input_embedding: shape (384,) minilm embedding of current input
        # current_turn: optional, used for recency weights in memory aggregation
        #   if none, all memories weighted by similarity alone
        #
        # returns shape (448,) = [z_graph | z_affect | z_memory | z_input]
        z_graph  = self._encode_graph(graph, input_embedding)
        z_affect = self._encode_affect(affect)
        z_memory = self._encode_memory(memory, input_embedding, current_turn)
        z_input  = self._encode_input(input_embedding)

        return np.concatenate([z_graph, z_affect, z_memory, z_input])

    # private encoders

    def _encode_graph(self, graph, input_embedding):
        # weighted mean of top-k concept embeddings -> R^128
        # if graph is empty, project a zero vector
        top_concepts = graph.top_concepts(n=self._top_k)

        if not top_concepts:
            agg = np.zeros(_SRC_EMBED, dtype=np.float32)
        else:
            # grab node weights from internal digraph
            g = graph._graph
            weights = np.array(
                [g.nodes[c].get("weight", 0.0) for c in top_concepts],
                dtype=np.float64,
            )
            # embed each concept string
            embeddings = np.stack(
                [self._embedder.embed(c) for c in top_concepts]
            )  # shape: (k, 384)

            # normalise weights to sum to 1
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones_like(weights) / len(weights)

            # weighted mean
            agg = (weights[:, None] * embeddings).sum(axis=0).astype(np.float32)

        return self._project(self.proj_graph, agg)

    def _encode_affect(self, affect):
        # identity-dimensioned projection of affect vector -> R^64
        state = affect._state.astype(np.float32)
        return self._project(self.proj_affect, state)

    def _encode_memory(self, memory, input_embedding, current_turn):
        # recency x similarity weighted mean of stored episode embeddings -> R^128
        #
        # queries chromadb directly to get all stored embeddings and metadata
        # then computes soft attention using:
        #   - retrieval similarity: cosine(episode_embed, input_embed)
        #   - exponential recency: exp(-lambda * |current_turn - episode_turn|)
        #
        # if memory is empty, project a zero vector
        count = memory._collection.count()
        if count == 0:
            agg = np.zeros(_SRC_EMBED, dtype=np.float32)
            return self._project(self.proj_memory, agg)

        # get all stored episode embeddings and metadata from chromadb
        all_data = memory._collection.get(
            include=["embeddings", "metadatas"],
        )
        embeddings_raw = all_data["embeddings"]    # list of lists
        metadatas = all_data["metadatas"]           # list of dicts

        episode_embeds = np.array(embeddings_raw, dtype=np.float32)  # (n, 384)

        # similarity weights
        inp = input_embedding.astype(np.float32)          # (384,)
        inp_norm = np.linalg.norm(inp)
        if inp_norm > 0:
            inp_unit = inp / inp_norm
        else:
            inp_unit = inp

        ep_norms = np.linalg.norm(episode_embeds, axis=1, keepdims=True)
        ep_norms = np.maximum(ep_norms, 1e-12)
        ep_unit = episode_embeds / ep_norms

        sim_weights = ep_unit @ inp_unit                  # (n,)
        # clamp negative similarities to zero, they dont carry useful signal
        sim_weights = np.maximum(sim_weights, 0.0)

        # recency weights
        if current_turn is not None:
            turns = np.array(
                [m.get("turn", 0) for m in metadatas], dtype=np.float64,
            )
            distances = np.abs(current_turn - turns)
            recency_weights = np.exp(-self._recency_decay * distances)
        else:
            recency_weights = np.ones(count, dtype=np.float64)

        # combined weights
        combined = sim_weights.astype(np.float64) * recency_weights
        w_sum = combined.sum()
        if w_sum > 0:
            combined = combined / w_sum
        else:
            combined = np.ones(count, dtype=np.float64) / count

        # weighted mean of episode embeddings
        agg = (combined[:, None] * episode_embeds).sum(axis=0).astype(np.float32)

        return self._project(self.proj_memory, agg)

    def _encode_input(self, input_embedding):
        # direct projection of minilm input embedding -> R^128
        inp = input_embedding.astype(np.float32)
        return self._project(self.proj_input, inp)

    # helpers

    @staticmethod
    def _project(layer, source):
        # pass a numpy vector through a pytorch linear layer, return numpy
        t = torch.from_numpy(source).unsqueeze(0)   # (1, src_dim)
        out = layer(t)                                # (1, tgt_dim)
        return out.squeeze(0).numpy()


# =====================
# standalone test
# =====================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("StateEncoder standalone integration test")
    print("=" * 60)

    # build dependencies
    embedder = Embedder()

    graph = ConceptGraph()
    graph.update(["transformer", "attention", "memory"], surprise=0.8)
    graph.update(["attention", "self-attention", "query"], surprise=0.5)
    graph.update(["memory", "retrieval", "hippocampus"], surprise=0.7)

    affect = AffectiveState()
    affect.update(surprise=0.9, sentiment=0.6)
    affect.update(surprise=0.4, sentiment=-0.3)

    mem = EpisodicMemory(embedder)
    mem.write("The transformer architecture uses self-attention.",
              {"turn": 1})
    mem.write("Episodic memory is crucial for continuity.",
              {"turn": 2})
    mem.write("Curiosity drives exploration under uncertainty.",
              {"turn": 3})

    input_text = "How does attention relate to memory retrieval?"
    input_emb = embedder.embed(input_text)

    # encode
    encoder = StateEncoder(embedder, top_k=10, recency_halflife=10.0)
    z = encoder.encode(graph, affect, mem, input_emb, current_turn=4)

    # assertions
    assert z.shape == (Z_DIM,), f"Expected shape ({Z_DIM},), got {z.shape}"
    print(f"  z_t shape        : {z.shape}  ok")

    z_graph  = z[:_TGT_GRAPH]
    z_affect = z[_TGT_GRAPH : _TGT_GRAPH + _TGT_AFFECT]
    z_memory = z[_TGT_GRAPH + _TGT_AFFECT : _TGT_GRAPH + _TGT_AFFECT + _TGT_MEMORY]
    z_input  = z[_TGT_GRAPH + _TGT_AFFECT + _TGT_MEMORY :]

    for name, vec in [("z_graph", z_graph), ("z_affect", z_affect),
                      ("z_memory", z_memory), ("z_input", z_input)]:
        norm = np.linalg.norm(vec)
        assert norm > 1e-6, f"{name} is near-zero (norm={norm:.2e})"
        print(f"  {name:12s} dim={vec.shape[0]:3d}  norm={norm:.4f}  ok")

    # verify projections are accessible for optimisation
    params = list(encoder.parameters())
    assert len(params) == 4, f"Expected 4 parameter tensors, got {len(params)}"
    total = sum(p.numel() for p in params)
    print(f"  trainable params : {total:,}")
    print(f"  all projections  : {[p.shape for p in params]}")

    # same input should give same output
    z2 = encoder.encode(graph, affect, mem, input_emb, current_turn=4)
    assert np.allclose(z, z2, atol=1e-6), "Determinism check failed"
    print(f"  determinism      : ok")

    # edge case: empty graph + empty memory
    empty_graph = ConceptGraph()
    empty_mem = EpisodicMemory(embedder)
    z_empty = encoder.encode(empty_graph, affect, empty_mem, input_emb,
                             current_turn=0)
    assert z_empty.shape == (Z_DIM,), "Shape mismatch on empty modules"
    # z_graph and z_memory should be zero when inputs are zero
    z_eg = z_empty[:_TGT_GRAPH]
    z_em = z_empty[_TGT_GRAPH + _TGT_AFFECT : _TGT_GRAPH + _TGT_AFFECT + _TGT_MEMORY]
    assert np.allclose(z_eg, 0.0, atol=1e-8), "Empty graph != zero"
    assert np.allclose(z_em, 0.0, atol=1e-8), "Empty memory != zero"
    print(f"  empty modules    : z_graph=0, z_memory=0  ok")

    print("-" * 60)
    print("All assertions passed.")
    print("=" * 60)
