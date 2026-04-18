# graph.py - concept graph for topos
#
# weighted directed graph of concepts seen during a session
# node weights and edge weights are accumulated surprise scores

from __future__ import annotations

import itertools
from typing import Optional

import networkx as nx

import config


class ConceptGraph:
    # weighted directed graph of concepts, pruned to MAX_GRAPH_NODES

    def __init__(self):
        self._graph = nx.DiGraph()

    def update(self, concepts, surprise):
        # takes a list of concept strings and a surprise value
        # adds/updates nodes and creates directed edges between all pairs
        # prunes if we go over the node limit
        if not concepts:
            return

        g = self._graph

        # update nodes
        for concept in concepts:
            if concept not in g:
                g.add_node(concept, weight=0.0)
            g.nodes[concept]["weight"] += surprise

        # directed edges between all pairs (no self loops)
        for src, dst in itertools.permutations(concepts, 2):
            if not g.has_edge(src, dst):
                g.add_edge(src, dst, weight=0.0)
            g[src][dst]["weight"] += surprise

        # chop if too many nodes
        if g.number_of_nodes() > config.MAX_GRAPH_NODES:
            self._prune_lowest_weight_node()

    def top_concepts(self, n=5):
        # returns the n highest weight concept nodes
        nodes = self._graph.nodes(data="weight", default=0.0)
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        return [name for name, _ in sorted_nodes[:n]]

    def related(self, concept, n=3):
        # returns the n strongest neighbors of a concept
        # empty list if concept isnt in the graph
        g = self._graph
        if concept not in g:
            return []

        # grab (neighbor, edge_weight) for outgoing edges
        neighbors = [
            (dst, data.get("weight", 0.0))
            for dst, data in g[concept].items()
        ]
        neighbors.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in neighbors[:n]]

    def summary(self):
        # plain english summary of top concepts
        top = self.top_concepts(n=5)
        if not top:
            return "Salient concepts: none"
        return f"Salient concepts: {', '.join(top)}"

    # internal stuff

    def _prune_lowest_weight_node(self):
        # remove the single weakest node
        nodes = self._graph.nodes(data="weight", default=0.0)
        lowest = min(nodes, key=lambda item: item[1])
        self._graph.remove_node(lowest[0])


# quick test
if __name__ == "__main__":
    cg = ConceptGraph()

    cg.update(["memory", "attention", "transformer"], surprise=0.8)
    cg.update(["attention", "self-attention", "query"], surprise=0.5)
    cg.update(["memory", "retrieval"], surprise=0.6)

    print("Top concepts:", cg.top_concepts(n=5))
    print("Related to 'attention':", cg.related("attention", n=3))
    print("Related to 'unknown':", cg.related("unknown"))
    print(cg.summary())
