# concept_graph_entropy_diagnostic.py - concept graph convergence diagnostic
#
# runs AFTER sequential priming and BEFORE --autonomous.
# reads a JSONL run log and computes:
#   - per-turn Shannon entropy over top-N concept weights
#   - Jaccard similarity between consecutive top-5 concept sets
#   - convergence verdict: CONVERGED / DYNAMIC / BORDERLINE
#
# usage:
#   python concept_graph_entropy_diagnostic.py --run-log runs/<logfile>.jsonl
#
# if top_5_concepts is present in the log, reads directly (no replay).
# if absent, falls back to replaying turns through a fresh GlobalWorkspace.

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# =====================
# tunable thresholds
# =====================

# number of recent turns to analyse
ANALYSIS_WINDOW = 20

# top-N concepts to pull from the graph when replaying (log-read path
# uses whatever N was logged, typically 5)
TOP_N = 5

# convergence thresholds
JACCARD_CONVERGED = 0.6    # mean Jaccard >= this => concepts are sticky
JACCARD_DYNAMIC   = 0.4    # mean Jaccard <  this => high churn
ENTROPY_STD_LOW   = 0.05   # std < this => weight distribution is frozen
ENTROPY_STD_HIGH  = 0.10   # std > this => weight distribution is shifting

# =====================
# threshold justifications (inline for reviewers)
# =====================
# JACCARD_CONVERGED = 0.6  — 3-of-5 overlap is the boundary where the top set
#                            is dominated by a fixed core; above this the graph
#                            is echoing itself.
# JACCARD_DYNAMIC   = 0.4  — below 2-of-5 overlap means concepts are churning
#                            enough that the workspace is not yet settled.
# ENTROPY_STD_LOW   = 0.05 — entropy fluctuation under 0.05 nats means the
#                            weight distribution is effectively static across
#                            turns.
# ENTROPY_STD_HIGH  = 0.10 — above 0.10 nats means the distribution is
#                            meaningfully reshaping turn-over-turn.


def shannon_entropy(weights):
    """Compute Shannon entropy over a list of non-negative weights.

    Normalises to a probability distribution first.
    Returns 0.0 for empty or all-zero inputs.
    """
    total = sum(weights)
    if total <= 0 or len(weights) == 0:
        return 0.0
    probs = [w / total for w in weights]
    return -sum(p * math.log(p) for p in probs if p > 0)


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 1.0


def load_log(path):
    """Load a JSONL log file, return list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def has_top5_field(records):
    """Check whether the log contains top_5_concepts per record."""
    if not records:
        return False
    # check the first record
    return "top_5_concepts" in records[0]


def get_turn_index(record):
    """Extract the turn index from a log record.

    Mega-longitudinal logs use 'turn_index'.
    Integration logs use 'turn'.
    """
    if "turn_index" in record:
        return record["turn_index"]
    return record.get("turn", 0)


# =====================
# primary path: read from log
# =====================

def analyse_from_log(records, window=ANALYSIS_WINDOW):
    """Extract per-turn entropy and top-5 sets from logged data.

    Uses top_5_concepts field directly.  To compute entropy we need
    weights, but the log only stores concept names (not weights).
    In this path we compute entropy over an *implied uniform distribution*
    since the log doesn't carry per-concept weights.

    However — if the graph module is importable we can do better: build
    the graph from concept co-occurrence in the log and read weights
    from it.  We prefer the replay fallback for weight-accurate entropy,
    but for the log-read path we accept the approximation.

    UPDATE: the replay fallback reconstructs the full graph, so we can
    actually extract real weights there.  For the log-read path, we use
    a lightweight replay *of just the concept graph* to get real weights.
    """
    # take last `window` turns
    tail = records[-window:]
    if len(tail) < window:
        print(f"WARNING: log has only {len(tail)} turns, fewer than "
              f"requested {window}. Analysing all available turns.",
              file=sys.stderr)

    # lightweight replay: build a ConceptGraph by feeding the logged
    # top_5 concept lists turn by turn over ALL records, so that the
    # graph state at any tail turn reflects cumulative history.
    #
    # import here to keep the script usable even if workspace deps
    # are missing (the pure-log path still works with uniform entropy)
    try:
        from graph import ConceptGraph
        import config as _cfg
        graph = ConceptGraph()
        graph_available = True
    except ImportError:
        graph = None
        graph_available = False

    per_turn = []

    # we need to know which records are in the tail
    tail_start_idx = len(records) - len(tail)

    for i, record in enumerate(records):
        top5 = record.get("top_5_concepts", [])

        # feed concepts into graph with uniform surprise=1.0
        # (we only care about accumulated weights, not about
        # reproducing exact surprise values)
        if graph_available:
            graph.decay()
            if top5:
                graph.update(top5, 1.0)

        # only record per-turn data for the tail
        if i >= tail_start_idx:
            turn_idx = get_turn_index(record)
            top5_set = frozenset(top5)

            if graph_available:
                # get real weights from the graph for the top-N nodes
                nodes = graph._graph.nodes(data="weight", default=0.0)
                sorted_nodes = sorted(nodes, key=lambda x: x[1],
                                      reverse=True)
                top_n_weights = [w for _, w in sorted_nodes[:TOP_N]]
                entropy = shannon_entropy(top_n_weights)
            else:
                # fallback: uniform entropy over however many concepts
                # the log recorded
                n = len(top5)
                entropy = math.log(n) if n > 0 else 0.0

            per_turn.append({
                "turn_index": turn_idx,
                "entropy": entropy,
                "top_5": list(top5),
            })

    return per_turn


# =====================
# fallback path: replay through GlobalWorkspace
# =====================

def analyse_from_replay(records, window=ANALYSIS_WINDOW):
    """Replay with a fresh GlobalWorkspace and extract real graph state.

    This is used when the log doesn't contain top_5_concepts.
    The log must have an 'input' field per record.
    """
    from workspace import GlobalWorkspace

    ws = GlobalWorkspace()

    tail = records[-window:]
    if len(tail) < window:
        print(f"WARNING: log has only {len(tail)} turns, fewer than "
              f"requested {window}. Analysing all available turns.",
              file=sys.stderr)

    tail_start_idx = len(records) - len(tail)
    per_turn = []

    for i, record in enumerate(records):
        # get the input text
        text = record.get("input")
        if text is None:
            # some logs might store it differently
            text = record.get("text", "")
        if not text:
            continue

        # run through process (full cognitive loop, no SLM call)
        ws.process(text)

        if i >= tail_start_idx:
            turn_idx = get_turn_index(record)

            # extract top-N with weights from the live graph
            nodes = ws.graph._graph.nodes(data="weight", default=0.0)
            sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
            top_n = sorted_nodes[:TOP_N]
            top5_names = [name for name, _ in top_n]
            top5_weights = [w for _, w in top_n]

            entropy = shannon_entropy(top5_weights)

            per_turn.append({
                "turn_index": turn_idx,
                "entropy": entropy,
                "top_5": top5_names,
            })

    return per_turn


# =====================
# compute diagnostics
# =====================

def compute_diagnostics(per_turn):
    """Compute summary statistics and verdict from per-turn data."""
    entropies = [t["entropy"] for t in per_turn]
    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    std_entropy = (
        (sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies))
        ** 0.5
        if entropies else 0.0
    )

    # Jaccard similarities between consecutive top-5 sets
    jaccards = []
    for i in range(1, len(per_turn)):
        set_a = frozenset(per_turn[i - 1]["top_5"])
        set_b = frozenset(per_turn[i]["top_5"])
        jaccards.append(jaccard(set_a, set_b))

    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
    std_jaccard = (
        (sum((j - mean_jaccard) ** 2 for j in jaccards) / len(jaccards))
        ** 0.5
        if jaccards else 0.0
    )

    # verdict
    if mean_jaccard >= JACCARD_CONVERGED and std_entropy < ENTROPY_STD_LOW:
        verdict = "CONVERGED"
    elif mean_jaccard < JACCARD_DYNAMIC or std_entropy > ENTROPY_STD_HIGH:
        verdict = "DYNAMIC"
    else:
        verdict = "BORDERLINE"

    return {
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "mean_jaccard": mean_jaccard,
        "std_jaccard": std_jaccard,
        "jaccards": jaccards,
        "verdict": verdict,
    }


VERDICT_MESSAGES = {
    "CONVERGED": (
        "High echo chamber risk. Consider extending priming or injecting "
        "external perturbation before autonomous phase."
    ),
    "DYNAMIC": (
        "Workspace has sufficient dynamic range. Proceed."
    ),
    "BORDERLINE": (
        "Marginal. Monitor epistemic signal closely in first 5 autonomous "
        "turns."
    ),
}


# =====================
# output
# =====================

def print_report(per_turn, diagnostics, log_path):
    """Print the diagnostic report to stdout."""
    print()
    print("=" * 70)
    print("CONCEPT GRAPH ENTROPY DIAGNOSTIC")
    print("=" * 70)
    print(f"  Log: {log_path}")
    print(f"  Turns analysed: {len(per_turn)}")
    print()

    # per-turn table
    print(f"  {'Turn':>5}  {'Entropy':>8}  {'Top 5 concepts'}")
    print(f"  {'-----':>5}  {'--------':>8}  {'-' * 40}")
    for t in per_turn:
        concepts_str = ", ".join(t["top_5"][:5])
        print(f"  {t['turn_index']:5d}  {t['entropy']:8.4f}  {concepts_str}")

    print()
    print(f"  Mean entropy:   {diagnostics['mean_entropy']:.4f}")
    print(f"  Std entropy:    {diagnostics['std_entropy']:.4f}")
    print()

    # Jaccard table
    print(f"  {'Turns':>12}  {'Jaccard':>8}")
    print(f"  {'-----':>12}  {'--------':>8}")
    for i, j_val in enumerate(diagnostics["jaccards"]):
        t_prev = per_turn[i]["turn_index"]
        t_curr = per_turn[i + 1]["turn_index"]
        print(f"  {t_prev:5d}→{t_curr:<5d}  {j_val:8.4f}")

    print()
    print(f"  Mean Jaccard:   {diagnostics['mean_jaccard']:.4f}")
    print(f"  Std Jaccard:    {diagnostics['std_jaccard']:.4f}")
    print()

    # verdict
    verdict = diagnostics["verdict"]
    print(f"  VERDICT: {verdict}")
    print(f"  → {VERDICT_MESSAGES[verdict]}")
    print("=" * 70)


def write_output(per_turn, diagnostics, log_path, out_path):
    """Write structured JSON output."""
    output = {
        "run_log": str(log_path),
        "turns_analysed": len(per_turn),
        "per_turn": per_turn,
        "mean_entropy": diagnostics["mean_entropy"],
        "std_entropy": diagnostics["std_entropy"],
        "mean_jaccard": diagnostics["mean_jaccard"],
        "std_jaccard": diagnostics["std_jaccard"],
        "verdict": diagnostics["verdict"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Output written to: {out_path}")


# =====================
# main
# =====================

def main():
    global TOP_N
    parser = argparse.ArgumentParser(
        description="Concept graph entropy diagnostic for TOPOS priming runs"
    )
    parser.add_argument(
        "--run-log", required=True, type=str,
        help="Path to the priming JSONL log file"
    )
    parser.add_argument(
        "--window", type=int, default=ANALYSIS_WINDOW,
        help=f"Number of recent turns to analyse (default: {ANALYSIS_WINDOW})"
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N,
        help=f"Number of top concepts for entropy (default: {TOP_N})"
    )
    parser.add_argument(
        "--end-turn", type=int, default=None,
        help="Select the window of turns ending at this index (inclusive)"
    )
    args = parser.parse_args()

    TOP_N = args.top_n

    log_path = Path(args.run_log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # load log
    records = load_log(log_path)
    if not records:
        print(f"ERROR: log file is empty: {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(records)} records from {log_path}")

    if args.end_turn is not None:
        filtered = []
        for r in records:
            filtered.append(r)
            if get_turn_index(r) == args.end_turn:
                break
        records = filtered
        if not records or get_turn_index(records[-1]) != args.end_turn:
            print(f"WARNING: turn {args.end_turn} not found in log.", file=sys.stderr)
        print(f"  Filtered to {len(records)} records ending at turn {args.end_turn}")

    # choose path
    if has_top5_field(records):
        print(f"  top_5_concepts field found — using log-read path")
        per_turn = analyse_from_log(records, window=args.window)
    else:
        print(f"  top_5_concepts field NOT found — using replay fallback")
        per_turn = analyse_from_replay(records, window=args.window)

    if not per_turn:
        print("ERROR: no turns to analyse after extraction.", file=sys.stderr)
        sys.exit(1)

    # compute
    diagnostics = compute_diagnostics(per_turn)

    # report
    print_report(per_turn, diagnostics, log_path)

    # write JSON output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("runs") / f"concept_entropy_diagnostic_{timestamp}.json"
    write_output(per_turn, diagnostics, log_path, out_path)


if __name__ == "__main__":
    main()
