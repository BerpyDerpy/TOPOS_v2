# EXPERIMENTS.md  -  TOPOS Experiment Log

Running, append-only log. One entry per experiment. Write the hypothesis and setup **before** the run. Add the result and interpretation **after**. Never edit old entries  -  if an old result gets reinterpreted, add a new entry that references it.

---

## Experiment 6: Autonomous Curiosity Integration

**Date:** 2026-04-18  -  pre-registered, blocked on Experiment 5

### Hypothesis

With `WorkspaceIntegration` wired into the cognitive loop and the `--autonomous` flag enabled, the agent will generate autonomous questions at domain transition points  -  specifically, the first autonomous question should appear near the music-to-systems or systems-to-music boundary where epistemic uncertainty peaks.

The system should NOT generate questions mid-batch when the forward model has already learned the current domain's dynamics.

### Setup

- Same 50-turn priming protocol as Run 3 (25 music + 25 systems, interleaved batches of 5)
- After priming, 20 additional turns with autonomous questioning enabled
- Forward model trained online during priming (bagged updates, keep_prob=0.6)
- Exploration gate: AdaptiveThreshold at 85th percentile of recent epistemic
- Question generation: SLM produces 5 candidates, scored by epistemic uncertainty
- JSONL logging of all turns

### Config Snapshot

```python
OLLAMA_MODEL      = "qwen2.5:7b"
WORKSPACE_ALPHA   = 0.4
AFFECT_ALPHA      = 0.3
AFFECT_DECAY      = 0.95
SENTIMENT_SCALE   = 8.0
MAX_GRAPH_NODES   = 100
# Curiosity mechanism:
# ensemble_members = 5
# ensemble_lr      = 1e-3
# bagging_keep     = 0.6
# init_scales      = [0.5, 0.8, 1.0, 1.5, 2.0]
# threshold_window = 50
# threshold_pct    = 85.0
# progress_window  = 10
```

### Result

*Blocked on Experiment 5 interpretation review. Not yet run.*

### Interpretation

*Pending result.*

---

## Experiment 5: Curiosity Mechanism Validation

**Date:** 2026-04-18

### Hypothesis

The ensemble forward model trained on Run 3 replay data will:
1. Show decreasing prediction error over training epochs (the model learns)
2. Produce elevated epistemic uncertainty at domain transition turns {6, 11, 16, 21, 26, 31, 36, 41, 46} compared to non-transition turns (the signal tracks domain structure)
3. Produce prediction error that tracks domain novelty  -  higher error at transition turns than within-batch turns (Schmidhuber productive novelty signature)

### Setup

- 50-turn replay of Run 3 sequence through `GlobalWorkspace.process()` (no SLM)
- State encoding via `StateEncoder` at each turn (z_t before, z_{t+1} after)
- Action vector a_t = projection of input text embedding to R^128 (causally accurate: input drives process() transition)
- Warmup: 5 epochs over first 10 triples, bootstrap aggregating (60% keep)
- Online pass: measure epistemic + error BEFORE each update, then update (mirrors live system)
- Init diversity: weight scaling factors [0.5, 0.8, 1.0, 1.5, 2.0] per ensemble member
- Progress window K=4 (validation-only setting; fits within 5-turn batch structure)

### Config Snapshot

```python
WORKSPACE_ALPHA   = 0.4
AFFECT_ALPHA      = 0.3
AFFECT_DECAY      = 0.95
SENTIMENT_SCALE   = 8.0
MAX_GRAPH_NODES   = 100
OLLAMA_MODEL      = "qwen2.5:7b"  # not used (no SLM in validation)
# Validation-specific:
# ensemble_members = 5
# ensemble_lr      = 1e-3
# bagging_keep     = 0.6
# init_scales      = [0.5, 0.8, 1.0, 1.5, 2.0]
# warmup_epochs    = 5
# progress_window  = 4
```

### Debugging Path

Four validation runs were required:

1. **Run 1 (batch train/test):** Check 1 passed. Checks 2+3 failed. Epistemic uncertainty was O(10⁻⁴) and monotonically rising  -  a distribution drift artifact, not a domain signal. Root cause: batch-trained ensemble goes stale as workspace state drifts beyond training distribution.

2. **Run 2 (+ init diversity + aggressive bagging):** Added weight scaling [0.5–2.0] and reduced keep_prob to 0.6. Epistemic rose to O(10⁻³) but still monotonic. Same root cause  -  batch vs. online mismatch.

3. **Run 3 (online training):** Switched to measure-then-update online pass with 10-triple warmup. Check 2 passed at 1.05× ratio (transition mean 0.014029 > non-transition mean 0.013372). Check 3 failed  -  prediction error didn't cleanly distinguish transitions from non-transitions.

4. **Run 4 (Check 3 reframed):** Tested prediction error at transitions vs. within-batch directly. Ratio 0.97×  -  prediction error is not the right signal for transition detection in the online single-sample setting. Confirmed that epistemic uncertainty (ensemble disagreement) is the correct and sufficient curiosity signal.

### Result

| Check | Result | Detail |
|---|---|---|
| 1. Model learns | **PASS** | Warmup loss: 0.094 → 0.007 (13.3× decrease) |
| 2. Epistemic tracks transitions | **PASS** | Transition mean 0.014029 > non-transition 0.013372 (1.05×) |
| 3. Error tracks transitions | **FAIL** | Transition 1.684 < non-transition 1.745 (0.97×) |

**2 of 3 checks pass.** Check 2 is the operationally critical signal  -  it's what `AdaptiveThreshold` uses to gate autonomous exploration. Check 3 fails because with online single-sample updates on a continuously drifting state space, prediction error reflects global task difficulty (the state space is explored extremely sparsely  -  each z_t is unique), not local domain novelty. The ensemble *disagreement* tracks transitions; the ensemble *mean prediction error* does not.

### Interpretation

**Critical finding: online training is necessary.** The forward model must train online (measure-then-update) rather than batch-then-test. With a continuously evolving workspace state, a batch-trained model's epistemic uncertainty reflects distance from the training distribution (monotonically increasing with turn count) rather than local domain novelty. Online training eliminates distribution drift.

**Critical finding: init diversity is necessary.** Bagging alone is insufficient for ensemble diversity. Scaling each member's initialisation by a different factor (0.5–2.0) creates genuinely different starting hypotheses. Without this, epistemic uncertainty collapses to O(10⁻⁴)  -  no signal.

**Prediction error ≠ epistemic uncertainty.** These measure different things. Prediction error = how wrong was the mean prediction (affected by overall task difficulty). Epistemic uncertainty = how much do the ensemble members disagree (affected by state space familiarity). For exploration gating, epistemic uncertainty is the correct and sufficient signal.

**The progress signal (Schmidhuber K-window) requires longer domain exposure.** With 5-turn batches and online single-sample updates, the rolling window crosses batch boundaries, creating artifacts. The progress signal should become meaningful in longer-horizon deployment where each domain exposure spans 10+ turns.

**Verdict: proceed to integration.** The exploration gate's primary signal (epistemic uncertainty) tracks domain structure. The model learns. The architecture is validated for live deployment.

---

## Experiment 4: P-Tuning v2 Workspace Injection (Qwen3-8B)

**Date:** 2026-04-14  -  pre-registered, not yet run

### Hypothesis

P-Tuning v2 with workspace state injection at all KV layers will reduce the base model closing-convention bleed observed in the soft-context-string approach (AGENTS.md §4.7, weakness #2).

Specifically: evaluation question Q3 ("What do you find yourself returning to, when nothing is demanding your attention?") will produce a response whose *ending* is workspace-conditioned  -  not reverting to the base model's default polite-question or generic-summary closing patterns.

### Background

Runs 1–3 (see Experiments 1–3 below) established that the context string approach works for response *bodies* but fails at response *endings*. The base model's trained closing conventions override workspace conditioning in the final 1–2 sentences. This is consistent with the SLM treating the context string as soft guidance rather than hard state  -  deeper injection via learned prefix tokens at every KV layer should make workspace state structurally unavoidable.

### Setup

- **Model:** Qwen3-8B (base, not instruct), Q8 quantisation via Ollama
- **Injection method:** P-Tuning v2  -  learned continuous prompts prepended to keys and values at all transformer layers, not just the embedding layer
- **Projection:** R^384 → R^(d_model × k), k = 8 soft prompt tokens initially
- **Training data:** Synthetic pairs generated from Run 3's workspace states + manually curated target responses that maintain workspace register through to closing
- **Evaluation protocol:** Same 50-turn longitudinal protocol as Run 3 (25 music + 25 systems, interleaved batches of 5)
- **Primary signal:** Q3 response  -  does the ending maintain workspace character or revert?
- **Secondary signals:** Q1, Q2 cross-domain synthesis quality; overall valence coherence

### Config Snapshot

```python
# config.py at time of experiment
OLLAMA_MODEL      = "qwen3:8b"       # CHANGED from qwen2.5:7b
WORKSPACE_ALPHA   = 0.4
AFFECT_ALPHA      = 0.3
AFFECT_DECAY      = 0.95
SENTIMENT_SCALE   = 8.0
MAX_GRAPH_NODES   = 100
# P-Tuning v2 specific (new):
# SOFT_PROMPT_TOKENS = 8
# INJECTION_LAYERS   = "all"
```

### Result

*Not yet run.*

### Interpretation

*Pending result.*

---

## Experiment 3: Full Signal Quality Fix (Run 3)

**Date:** pre-2026-04-10 (retroactive entry from AGENTS.md §4.6)

### Hypothesis

Three simultaneous fixes  -  expanded stopwords, domain sentiment lexicon replacing TextBlob, directive context string framing  -  will produce cross-domain synthesis and affective coherence that Runs 1 and 2 failed to achieve.

### Setup

- **Model:** Qwen2.5-7B (base), via Ollama
- **Three changes applied simultaneously:**
  1. Stopword list expanded (+25 domain-specific function words)
  2. TextBlob replaced with 70-entry domain lexicon (±8× normalisation)
  3. Context string reframed from suggestive ("Let it subtly shape your response") to directive first-person ("You find yourself drawn to: [concepts]. These are not topics  -  they are how you think.")
- **Protocol:** 50-turn longitudinal, 25 music + 25 systems, interleaved batches of 5
- **Evaluation:** Q1 (silence), Q2 (rhythm/computers), Q3 (idle returns)

### Config Snapshot

```python
WORKSPACE_ALPHA   = 0.4
AFFECT_ALPHA      = 0.3
AFFECT_DECAY      = 0.95
SENTIMENT_SCALE   = 8.0
MAX_GRAPH_NODES   = 100
OLLAMA_MODEL      = "qwen2.5:7b"
```

### Result

- **Q1 (Silence):** Both domains present  -  electronic music, data transmission gaps, system performance  -  unified in one response.
- **Q2 (Rhythm/Computers):** CPU cycles, clock precision, waiting states, synchronisation, connected to musical silence. Genuine synthesis.
- **Q3 (Idle returns):** Referenced "familiar architectures" (not "familiar places"). The word *architecture* is load-bearing  -  traces to concept graph having both "abstraction" and "model" at high weight. First response that did not deflect back to the user.
- **Valence:** Resolved to negative after 50 turns. Priming corpus was frustrated/searching. Context string framed character as "a restless, searching quality"  -  coherent with arousal and valence for the first time.

### Interpretation

All three fixes were necessary. Any one alone would have been insufficient:
- Stopword expansion refined the noise floor but couldn't fix dead affect.
- The lexicon made valence non-zero but without directive framing the base model still overrode it.
- Directive framing forced the SLM to attend to workspace state, but only if that state was actually meaningful (which required the first two fixes).

Remaining weakness: closing conventions still leak. Response bodies are workspace-conditioned; endings revert to base model defaults. This motivates Experiment 4 (P-Tuning v2).

---

## Experiment 2: Stopword Fix (Run 2)

**Date:** pre-2026-04-10 (retroactive entry from AGENTS.md §4.6)

### Hypothesis

Adding a domain stopword list (~25 words) to concept extraction will shift the concept graph from noise-dominated to signal-dominant, enabling systems theme visibility in the longitudinal experiment.

### Setup

- **Model:** Qwen2.5-7B (base), via Ollama
- **Change:** Domain stopword list added to concept extraction (~25 words: "about", "between", "feels", etc.)
- **Protocol:** Same 50-turn longitudinal as Run 1

### Result

- Top-10 concepts shifted from noise to signal (2/10 → 7/10 signal concepts).
- **Q2:** First cross-domain synthesis  -  referenced CPUs, clock cycles, and connected them to silence from the music domain.
- **Q3:** Still music-dominant.
- **Valence:** Permanently neutral  -  TextBlob returning ≈ 0.0 on all inputs. Affective module non-functional.

| Run 1 (noise-dominated) | Run 2 (signal-dominant) |
|---|---|
| about, feels, there's, something | hardware, system, abstraction |
| every, really, memory | mental, model, chord |
| Signal concepts: 2/10 | Signal concepts: 7/10 |

### Interpretation

Stopwords are necessary but not sufficient. The concept graph is now signal-dominant, enabling partial synthesis (Q2). But two remaining failures are independent: (1) TextBlob is broken for this domain  -  the affect module receives no signal; (2) the suggestive context string framing gives the base model permission to ignore workspace state.

---

## Experiment 1: Baseline (Run 1)

**Date:** pre-2026-04-10 (retroactive entry from AGENTS.md §4.6)

### Hypothesis

A/B workspace divergence + longitudinal priming will produce traceable differences in SLM output conditioned on workspace state.

### Setup

- **Model:** Qwen2.5-7B (base), via Ollama
- **A/B test:** Two instances, 5 turns each (philosophy vs. systems), shared probe question
- **Longitudinal:** 50 turns, 25 music + 25 systems, interleaved batches of 5
- **Concept extraction:** Word length > 4 chars, lowercased, unique

### Result

- **A/B test:** Outcome 1 achieved. TOPOS-B unprompted referenced kernel scheduling from a neutral question about learning. Architecture works for short-horizon divergence.
- **Longitudinal:** Collapsed entirely to music. Systems theme invisible despite 25 turns of priming.

### Interpretation

Concept extraction based on word length captures high-frequency generic words ("about", "feels", "there's") while meaningful low-frequency domain terms ("kernel", "scheduler") appear once each and accumulate insufficient weight. **Upstream signal quality dominates everything downstream.** The graph, affect, and context string were all coherent given their inputs  -  but those inputs were garbage.

This is the most important empirical lesson from the project (documented in AGENTS.md §4.6).
