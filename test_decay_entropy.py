# test_decay_entropy.py - 30-turn decay validation
#
# replays 30 turns from the music corpus through GlobalWorkspace.process()
# (no SLM call) and prints per-turn Shannon entropy over top-5 concept
# weights, plus Jaccard similarity between consecutive top-5 sets.
#
# pass criteria:
#   - entropy std > 0  (weight distribution is not frozen)
#   - mean Jaccard < 0.9  (top-5 concepts are actually changing)
#
# usage:
#   python test_decay_entropy.py

import json
import math
import sys
from pathlib import Path

import numpy as np

import config
from workspace import GlobalWorkspace


# ---- settings ----
N_TURNS = 250
TOP_N   = 5
CORPUS  = "corpus/music_500.jsonl"


def shannon_entropy(weights):
    """Shannon entropy over non-negative weights (normalised to probs)."""
    total = sum(weights)
    if total <= 0 or len(weights) == 0:
        return 0.0
    probs = [w / total for w in weights]
    return -sum(p * math.log(p) for p in probs if p > 0)


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def load_corpus(path, n):
    """Load the first n turns from a JSONL corpus file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if len(entries) >= n:
                break
    return entries


def main():
    corpus_path = Path(CORPUS)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_corpus(corpus_path, N_TURNS)
    if len(entries) < N_TURNS:
        print(f"WARNING: corpus has only {len(entries)} turns, "
              f"wanted {N_TURNS}.", file=sys.stderr)

    print("=" * 70)
    print(f"DECAY ENTROPY DIAGNOSTIC — {len(entries)} turns, "
          f"GRAPH_DECAY={config.GRAPH_DECAY}")
    print("=" * 70)
    print()

    ws = GlobalWorkspace()

    per_turn = []  # list of dicts: turn, entropy, top5_names, top5_weights

    for i, entry in enumerate(entries):
        text = entry["text"]
        ws.process(text)

        # extract top-N with real weights from the live graph
        nodes = ws.graph._graph.nodes(data="weight", default=0.0)
        sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        top_n = sorted_nodes[:TOP_N]
        names = [name for name, _ in top_n]
        weights = [w for _, w in top_n]

        entropy = shannon_entropy(weights)

        per_turn.append({
            "turn": i + 1,
            "entropy": entropy,
            "top5": names,
            "top5_weights": weights,
        })

    # ---- per-turn table ----
    print(f"  {'Turn':>5}  {'Entropy':>8}  {'Top 5 concepts'}")
    print(f"  {'-----':>5}  {'--------':>8}  {'-' * 50}")
    for t in per_turn:
        concepts_str = ", ".join(t["top5"][:5])
        print(f"  {t['turn']:5d}  {t['entropy']:8.4f}  {concepts_str}")

    # ---- entropy stats ----
    entropies = [t["entropy"] for t in per_turn]
    mean_e = sum(entropies) / len(entropies)
    std_e = (sum((e - mean_e) ** 2 for e in entropies) / len(entropies)) ** 0.5

    print()
    print(f"  Mean entropy:  {mean_e:.4f}")
    print(f"  Std entropy:   {std_e:.4f}")

    # ---- Jaccard between consecutive top-5 sets ----
    jaccards = []
    print()
    print(f"  {'Turns':>12}  {'Jaccard':>8}")
    print(f"  {'-----':>12}  {'--------':>8}")
    for i in range(1, len(per_turn)):
        set_a = frozenset(per_turn[i - 1]["top5"])
        set_b = frozenset(per_turn[i]["top5"])
        j = jaccard(set_a, set_b)
        jaccards.append(j)
        print(f"  {per_turn[i-1]['turn']:5d}→{per_turn[i]['turn']:<5d}  {j:8.4f}")

    mean_j = sum(jaccards) / len(jaccards) if jaccards else 0.0

    print()
    print(f"  Mean Jaccard:  {mean_j:.4f}")

    # ---- verdict ----
    print()
    print("=" * 70)
    passed = True
    if std_e <= 0:
        print("  FAIL: entropy std is zero — weight distribution is frozen")
        passed = False
    else:
        print(f"  PASS: entropy std = {std_e:.4f} (nonzero)")

    if mean_j >= 0.9:
        print(f"  FAIL: mean Jaccard = {mean_j:.4f} (>= 0.9 — top-5 locked)")
        passed = False
    else:
        print(f"  PASS: mean Jaccard = {mean_j:.4f} (< 0.9 — concepts shifting)")

    print()
    if passed:
        print("  VERDICT: PASS — decay is restoring concept discrimination")
    else:
        print("  VERDICT: FAIL — decay insufficient or not applied")
    print("=" * 70)


if __name__ == "__main__":
    main()
