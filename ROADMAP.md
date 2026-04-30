# ROADMAP.md - TOPOS Planning Surface

Living document. Rewrite freely.

*Last updated: 2026-04-29*

---

## Current State

Experiment 6 completed. Both hypotheses failed. The core finding: the SLM is an unreliable measurement instrument for workspace state. Curiosity (if present) lives in the workspace. The mouth does not faithfully express the mind.

The workspace mechanic itself is validated: epistemic uncertainty spikes 2.41x at domain transitions (Exp 7), the concept graph evolves with content, affect accumulates. These are genuine internal state changes.

The behavioral layer on top is where the problems are: autonomous concept selection is round-robin, affect has no effect on decisions, the exploration gate misfires on cold-start artifacts, and echo-chamber feedback accelerates attractor convergence.

---

## Priority 1: Direct Workspace Measurement

Before running more SLM-dependent experiments, build the ability to observe the workspace state directly and continuously.

**What this means:**
- Real-time visualisation of concept graph topology (node weights, edges, churn over time)
- Affect trajectory plot (arousal + valence per turn)
- Epistemic uncertainty curve overlaid with domain labels
- Entropy of concept distribution per turn

The demo server (`demo_server.py`) is a starting point. Needs a live concept graph render that updates per turn, not just post-hoc.

**Why now:** Every future experiment result will be ambiguous until we can observe the mind independently of the mouth.

---

## Priority 2: Drive-Weighted Concept Selection

Replace round-robin autonomous concept selection with curiosity-weighted selection.

Current code (`integration.py`):
```python
query_concept = concepts[self._turn_index % len(concepts)]
```

Target: select the concept the forward model is most uncertain about. Requires embedding each top-k concept, computing predicted epistemic for each, and sampling from that distribution.

This is the most direct path toward genuine self-direction during idle turns.

---

## Priority 3: Affect Integration into Decisions

Arousal and valence are computed every turn but influence nothing. Two concrete connections to make:

1. **Arousal-gated memory writes.** `SURPRISE_THRESHOLD=0.35` is already defined in config but unused. High-arousal turns should write to long-term memory; low-arousal turns should not. This reduces memory noise and gives the memory module behavioural meaning.
2. **Valence-weighted concept selection.** During autonomous exploration, prefer concepts with positive valence associations (if the affect vector encodes that structure). Avoid concepts associated with negative valence saturation.

---

## Priority 4: Affect Reset Mechanism

Arousal saturated at 1.0 for all 520 turns of Exp 6. The EMA decay of 0.95 is too slow for 500-turn priming. The SLM saw identical affect values every turn.

Options:
- Faster decay rate during corpus priming (separate config for priming vs. autonomous phases)
- Hard reset of affect vector at phase boundaries
- Dynamic decay rate proportional to variance of recent surprise values (low variance = boring, increase decay to allow affect to shift)

---

## Priority 5: P-Tuning v2 (Exp 4)

Replace the natural-language context string with learned continuous prompts at all KV layers. Prescribed fix for base model closing-convention bleed.

Blocked on: training data generation from Run 3 workspace states, Qwen3-8B setup.

Depends on: Priority 1 (direct workspace measurement) to verify the projection is working.

---

## Priority 6: Perturbation Resistance

50-turn two-domain priming, then introduce a third domain at turn 50. Continue to 100+. Does accumulated identity shift gracefully or collapse?

Blocked on: P-Tuning v2 results. Not worth testing identity persistence if the SLM ignores workspace state at response endings.

---

## Parked (Don't Lose These)

**Ensemble diversity maintenance.** Weight norms converge over long runs (Exp 7 H4). No mechanism restores diversity once lost. Options: periodic re-initialisation of weakest member, diversity regularisation loss term, periodically sample weight perturbations from the init distribution.

**Cross-attention adapter.** Instead of P-Tuning v2's prefix injection, add cross-attention where the SLM attends to workspace state as a separate sequence. More expressive. Consider if P-Tuning v2 hits a ceiling.

**Dormant-active cycles.** Between sessions: decay low-weight nodes, prune weak edges, let affect settle toward zero. Simulate consolidation. Requires workspace serialisation (graph + affect + memory persistence) which is not implemented.

**Multi-agent workspace sharing.** Workspace is a serialisable data structure. Multiple SLMs could read from a shared workspace. Not a current priority.

**Introspection module.** Give the system read access to its own workspace state as a first-class reasoning object. Would require the SLM to output structured queries against graph/memory/affect.

---

## Known Blockers

| Blocker | Affects |
|---|---|
| No workspace persistence (in-memory only) | Dormant-active cycles, cross-session identity |
| Affect saturates on long runs (EMA too slow) | All experiments with 200+ priming turns |
| Round-robin concept selection | Autonomous exploration quality |
| Affect disconnected from decisions | Arousal-gated memory, valence-weighted exploration |
| Init diversity decays silently | Long-run epistemic signal quality |
