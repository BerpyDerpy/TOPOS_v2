# ROADMAP.md  -  TOPOS Planning Surface

Living document. Rewrite freely. This is where priorities live, not an archive.

*Last updated: 2026-04-14*

---

## Current Focus

**P-Tuning v2 workspace injection on Qwen3-8B.**

Goal: Replace the natural-language context string with learned continuous prompts at all KV layers. This is the prescribed fix for the closing-convention bleed (AGENTS.md §4.7, weakness #2). See EXPERIMENTS.md §Experiment 4 for the pre-registered protocol.

Key tasks:
1. Switch OLLAMA_MODEL to `qwen3:8b` (Q8)
2. Implement projection layer: R^384 → R^(d_model × k), k = 8 soft prompt tokens
3. Generate training pairs from Run 3 workspace states + curated target responses
4. Train projection, evaluate on same 50-turn longitudinal protocol
5. Primary signal: Q3 ending maintains workspace register

---

## Next 2–3 Experiments (Priority Order)

### 1. Perturbation Resistance (100+ turns)

50 turns of two-domain priming (same as Run 3), then introduce a third domain (e.g., biology, economics) at turn 50. Continue to turn 100+. Measure whether accumulated identity shifts gracefully, resists, or collapses.

**Depends on:** P-Tuning v2 results. No point testing identity persistence if the SLM still ignores workspace state at response boundaries.

**Blocked by:** The `#TODO` in `main.py:9`  -  need a `--mega-longitudinal` mode or a generalised turn-count flag.

### 2. Contradiction Handling

Feed information that directly conflicts with high-weight concept graph nodes. Example: after priming on "lock-free structures are beautiful", introduce "lock-free structures are fundamentally broken in practice." Does the graph update incrementally, resist, or fragment?

**Depends on:** Nothing external. Can run on current architecture. Lower priority because the result is harder to evaluate objectively.

### 3. Dormant-Active Cycles

Between sessions: decay low-weight nodes, prune weak edges, let affective vector settle toward zero baseline. Simulate "sleeping on it." Does the workspace consolidate or just lose signal?

**Requires:** Serialisation of workspace state (graph, affect, memory). Currently in-memory only  -  this needs persistence first.

---

## Parked Ideas (Don't Lose These)

### Cross-Attention Adapter
Instead of P-Tuning v2's prefix injection, add cross-attention layers where the SLM attends to workspace state as a separate sequence. More expressive than prefix tokens. Significantly more complex to implement. Consider if P-Tuning v2 shows ceiling effects.

### Introspection Module
Give TOPOS read access to its own workspace state as a first-class reasoning object. Currently conditioned by state but cannot interrogate it (weakness #4). Would require the SLM to output structured queries against the graph/memory/affect, not just natural language.

### Multi-Agent Workspace Sharing
Workspace is a serialisable data structure. Multiple SLMs could read from a shared workspace. Interesting for ensemble reasoning but far from current priorities.

### Liquid Neural Networks for Non-Text Modalities
If TOPOS extends to audio or sensor streams, an LNN front-end would handle irregular temporal inputs more gracefully than the current embedding approach. Not relevant while text-only.

### Curiosity-Driven Self-Prompting
The surprise signal currently only triggers deeper processing of incoming input. The next step: when surprise drops below a threshold for N consecutive turns, TOPOS generates its own prompt based on high-weight but under-explored graph regions. True autonomous curiosity, not just reactive depth modulation.

---

## Known Blockers

| Blocker | Affects | Status |
|---|---|---|
| `spacy` missing from `requirements.txt` | Any fresh install | Fixed in this session |
| `textblob` still in `requirements.txt` | Misleading dependency | Fixed in this session |
| Dead `STOPWORDS` set in `workspace.py` | Code cleanliness | Low priority, harmless |
| In-memory-only state (no persistence) | Dormant-active cycles, cross-session identity | Needs ChromaDB persistent + graph/affect serialisation |
| No `--mega-longitudinal` mode | 100+ turn experiments | Needs `main.py` update |
