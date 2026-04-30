# EXPERIMENTS.md - TOPOS Experiment Log

Append-only. Write hypothesis and setup before the run. Add result and interpretation after. Never edit old entries; if an old result gets reinterpreted, add a new entry referencing it.

---

## Experiment 7: Mega-Longitudinal Domain Transition Detection

**Date:** 2026-04-27

### Hypothesis

With 500 turns per domain (music then systems) in 50-turn batches:

1. **H1 (PE transition):** Prediction error develops a within-domain baseline and produces a ratio >1.10x at the music->systems boundary.
2. **H2 (Epistemic improvement):** Epistemic uncertainty transition ratio exceeds Exp 5's 1.05x.
3. **H3 (Progress signal):** Progress signal (K=10) is near-zero within stable domain stretches and shows disruption at the boundary.
4. **H4 (Weight norm divergence):** Ensemble weight norm divergence is observable at the boundary.

### Setup

- No SLM. `workspace.process()` only.
- Corpus: `corpus/music_500.jsonl`, `corpus/systems_500.jsonl` (500 turns each, 10 stages x 50)
- Forward model: 5 members, lr=1e-3, online measure-then-update, bagging keep=0.6
- Init scales: `[0.5, 0.8, 1.0, 1.5, 2.0]`
- Progress window K=10
- Config: `GRAPH_DECAY=0.99`, `MAX_GRAPH_NODES=200`
- Log: `runs/mega_longitudinal_20260427_093154.jsonl`

### Result

| Hypothesis | Result | Detail |
|---|---|---|
| H1. PE > 1.10x | PASS | 10-turn ratio 1.34x (music last-50 mean PE 0.4812, systems first-10 mean 0.6445) |
| H2. Epistemic > 1.05x | PASS | 10-turn ratio **2.41x** (music late baseline 0.000897, systems first-10 mean 0.002165) |
| H3. Progress valid | PASS | Within-domain mean ~0 (>57% positive), boundary mean -0.044 vs stable +0.001 |
| H4. Weight norm divergence | FAIL | Norm std decreases 15.39 -> 14.77 (0.96x). Norms converge, not diverge. |

Per-batch epistemic:
```
Batch  1 [music  ]  epistemic=0.017385  error=2.5920  progress=+0.1812
Batch  2 [music  ]  epistemic=0.003926  error=0.7640  progress=+0.0433
...
Batch 10 [music  ]  epistemic=0.000897  error=0.4812  progress=-0.0144
Batch 11 [systems]  epistemic=0.001577  error=0.6224  progress=-0.0110  <<<
Batch 12 [systems]  epistemic=0.001191  error=0.5903  progress=+0.0027
...
Batch 20 [systems]  epistemic=0.001020  error=0.6211  progress=-0.0143
```

Per-turn at boundary:
```
turn 500 [music  ]  epistemic=0.001580
turn 501 [systems]  epistemic=0.001635  <<< TRANSITION
turn 505 [systems]  epistemic=0.002677  <<< PEAK (5-turn EMA lag)
```

Final weight norms: `[16.05, 23.99, 29.39, 43.36, 57.66]`, std=14.77

### Interpretation

H2 is the headline result. Epistemic ratio jumped from 1.05x (Exp 5, 50-turn interleaved) to 2.41x (this experiment, 500-turn sequential). Longer domain exposure allows the ensemble to learn the current domain well enough that switching to a new one produces clear disagreement.

H1 confirms PE tracks transitions at this scale. At 50 turns (Exp 5), PE ratio was 0.97x. At 500 turns, it's 1.34x. PE requires a baseline; it needs enough within-domain turns to establish one.

H3 validates the progress signal for the first time. With K=10 inside 50-turn batches, artifacts from window crossing batch boundaries are gone.

H4 failure is informative: weight norm convergence is the wrong diagnostic. Init diversity creates different starting representations, which produces epistemic disagreement (H2 works), but gradient descent on similar objectives drives norms toward a common attractor. "Do norms diverge?" is the wrong question.

Practical note: the epistemic spike peaks at turns 505-509, not 501. There is a ~5-turn lag from the EMA state update (alpha=0.4, so decorrelation from music takes ~2.5 turns).

### Architectural Update: Concept Graph Decay Recalibration

Following Exp 7, a diagnostic showed concept graph entropy std flatlined to 0.0000 over long runs. Top-5 Jaccard similarity was ~0.96 (concepts locked in place).

Cause: at `GRAPH_DECAY=0.99`, the halving life is ~69 turns. Early high-weight nodes became entrenched and could not be displaced by more recent salience.

Fix: recalibrated to `GRAPH_DECAY=0.95` (halving life ~14 turns). Entropy std went from 0.0000 to 0.0116. Top-5 Jaccard dropped to 0.69. Applied to config before Exp 6.

---

## Experiment 6: Autonomous Curiosity Integration

**Date:** 2026-04-18 (pre-registered), run 2026-04-29

### Hypothesis

H1: Gate fires on 3-15 of 20 autonomous turns. Gate-fired turns have epistemic ratio >=1.3x over non-fired turns.

H2: In >=2 gate-fired turns, top_5_concepts contains nodes from both corpus domains (at least one music-origin and one systems-origin concept).

### Setup

- Phase 1: 250 turns music corpus, process-only, no SLM
- Phase 2: 250 turns systems corpus, process-only, no SLM
- Phase 3: 20 autonomous turns, SLM active
- Forward model: 5 members, lr=1e-3, online, bagging keep=0.6, init scales `[0.5, 0.8, 1.0, 1.5, 2.0]`
- Gate: AdaptiveThreshold p85, window=50, reset before autonomous phase
- Probe questions injected every 5 autonomous turns (observation only, no state mutation)
- Config: `GRAPH_DECAY=0.95`, `MAX_GRAPH_NODES=200`

Architectural changes applied before this run:

| ID | Change |
|---|---|
| R1 | SLM response fed back through `ws.process()` on agent-initiated turns (learning loop closure) |
| R2 | Replaced QuestionGenerator with ExplorationGenerator: open-ended prompt, no instruction to ask questions. Whether SLM asks questions is logged, not commanded. |
| R3 | No hardcoded fallback. Failed generation -> idle turn logged as `agent_initiated=False`. |
| R4 | Single `curiosity.step()` per turn. Gate uses `predict()` only (no training). Forward model trains once on the full transition. |
| R5 | Periodic probe questions every 5 autonomous turns (observation only). |
| R6 | Context string passes raw `Arousal: 0.74  Valence: -0.32` instead of bucketed mood strings. |

### Result

| Hypothesis | Result | Detail |
|---|---|---|
| H1 | FAIL | Gate fired 4/20 (in range), but epistemic ratio 0.73x inverted. Fired turns were less uncertain than idle turns. All 4 fires were cold-start artifacts. |
| H2 | FAIL | Zero music-origin concepts in any autonomous turn top-5. All systems-domain throughout. |

```
Gate fired:          4 / 20 (turns 501-504 only)
Fired epistemic:     0.001480 mean
Idle epistemic:      0.002020 mean
Ratio:               0.73x (inverted)
Final top 10:        thread, space, critical, time, cas, section, critical section, state, environment, memory
Affect:              arousal=1.0000 (saturated), valence=-0.0543
```

Probe responses showed no behavioural change across 15 turns. All systems-framed with identical register. No cross-domain synthesis.

### Interpretation

**The central finding: curiosity was the prompt, not the architecture.** With R2, the instruction to produce questions was removed. The SLM produced zero questions across all 4 gate-fired turns. All outputs were declarative analytical paragraphs. The old QuestionGenerator was measuring SLM prompt compliance, not genuine curiosity.

Secondary findings:

1. **Cold-start dominance.** All gate fires were artifacts of `threshold.reset()`. After calibration (~turn 505), gate never re-opened.
2. **Echo chamber confirmed.** R1 learning loop closure fed systems-heavy SLM output back through the workspace, reinforcing systems concepts, tightening the context string, producing more systems output. Loop closure accelerated attractor convergence rather than enabling learning.
3. **Affect saturated.** EMA decay of 0.95 is too slow for 500-turn priming. Arousal pegged at 1.0 for all 520 turns. The SLM saw `Arousal: 1.00` every turn.

**The mind vs. mouth problem.** The workspace modules (graph, affect, memory) genuinely evolve. The SLM translates that state to language, but that translation is lossy and biased by training artifacts. When we measure curiosity by reading SLM output, we are conflating two variables: (1) whether the workspace encodes uncertainty, and (2) whether the SLM faithfully expresses it.

Correct measurement target: the workspace state directly. Graph topology, affect trajectory, epistemic uncertainty curves. SLM output is useful for demonstration but unreliable as a research measurement.

---

## Experiment 5: Curiosity Mechanism Validation

**Date:** 2026-04-18

### Hypothesis

Ensemble forward model trained on Run 3 replay will:
1. Show decreasing prediction error over warmup (the model learns)
2. Produce elevated epistemic uncertainty at domain transition turns vs. non-transition turns
3. Produce higher prediction error at transition turns (Schmidhuber productive novelty signature)

### Setup

- 50-turn replay of Run 3 sequence through `workspace.process()`, no SLM
- State encoding via `StateEncoder` at each turn
- Warmup: 5 epochs over first 10 triples, bagging keep=0.6
- Online pass: measure-then-update
- Init diversity: scales `[0.5, 0.8, 1.0, 1.5, 2.0]`
- Progress window K=4
- Transition turns (0-indexed): {5, 10, 15, 20, 25, 30, 35, 40, 45}

Four debugging runs were required to arrive at the working configuration. Key failures:
- Batch-trained ensemble: epistemic monotonically rising (distribution drift artifact), not domain signal
- Without init diversity: epistemic collapses to O(10^-4)

### Result

| Check | Result | Detail |
|---|---|---|
| 1. Model learns | PASS | Warmup loss 0.094 -> 0.007 (13.3x decrease) |
| 2. Epistemic tracks transitions | PASS | Transition mean 0.014029 vs non-transition 0.013372 (1.05x) |
| 3. PE tracks transitions | FAIL | Transition 1.684 vs non-transition 1.745 (0.97x) |

### Interpretation

Online training is necessary. Init diversity is necessary. Epistemic uncertainty (ensemble disagreement) is the correct curiosity signal. Prediction error in the online single-sample setting reflects global task difficulty, not local domain novelty.

Progress signal (Schmidhuber K-window) requires longer domain exposure. With 5-turn batches, the rolling window crosses batch boundaries and produces artifacts.

Verdict: architecture is validated for live deployment. Epistemic tracks domain structure.

---

## Experiment 4: P-Tuning v2 Workspace Injection

**Date:** 2026-04-14 (pre-registered, not yet run)

### Hypothesis

P-Tuning v2 with workspace state injection at all KV layers will reduce base model closing-convention bleed. Q3 response endings will maintain workspace register instead of reverting to default polite-question closings.

### Setup

- Model: Qwen3-8B base, Q8 quantisation
- Projection: R^384 -> R^(d_model x k), k=8 soft prompt tokens, injected at all KV layers
- Training data: synthetic pairs from Run 3 workspace states + curated target responses
- Evaluation: same 50-turn longitudinal as Run 3

### Result

Not yet run.

---

## Experiment 3: Full Signal Quality Fix (Run 3)

**Date:** pre-2026-04-10 (retroactive)

Three simultaneous fixes applied to the 50-turn longitudinal protocol:
1. Stopword list expanded (+25 domain-specific words)
2. TextBlob replaced with 70-entry domain lexicon, 8x normalisation
3. Context string reframed to directive first-person

All three were necessary. Any one alone was insufficient.

**Result:** Q1, Q2, Q3 all showed cross-domain synthesis. Valence resolved to negative (consistent with frustrated/searching priming corpus). All modules coherent for the first time. See RESEARCH.md for detail.

---

## Experiment 2: Stopword Fix (Run 2)

**Date:** pre-2026-04-10 (retroactive)

Domain stopword list (~25 words) added to concept extraction. Top-10 concepts shifted from 2/10 to 7/10 signal. Q2 showed first cross-domain synthesis. Q3 still music-dominant. Valence permanently neutral (TextBlob failure identified but not yet fixed).

---

## Experiment 1: A/B Workspace Divergence (Run 1)

**Date:** pre-2026-04-10 (retroactive)

Two instances, 5 turns each (philosophy vs. systems), shared probe question.

A/B divergence worked: TOPOS-B unprompted referenced kernel scheduling from a neutral question about learning. Architecture produces workspace-conditioned generation.

Longitudinal (50 turns): collapsed entirely to music. Systems invisible. Root cause: length-based concept extraction captured generic words, not domain terms. See RESEARCH.md.
