# TOPOS Research Findings

## Core Problem

Build a conversational agent that learns from experience and exhibits curiosity as an intrinsic mechanism, without reinforcement learning, without a hardcoded persona, and without catastrophic forgetting.

Three sub-problems:
- **Continuous learning without catastrophic forgetting.** Standard fine-tuning overwrites previous knowledge. RL is unstable and requires a reward signal that is itself an unsolved problem for open-ended conversation.
- **Personality without programming.** A persona in a system prompt is a behavioural constraint, not a personality. Real personality is an emergent property of accumulated experience.
- **Curiosity without prompt.** A system that only responds is not curious. Curiosity requires an internal mechanism that initiates exploration independently.

---

## Architecture Selection

### Why RL Was Rejected

1. Catastrophic forgetting: gradient updates overwrite prior associations, especially severe below 10B parameters.
2. Instability: policy gradient methods require careful reward shaping. Defining a reward signal for open-ended conversation is itself unsolved.
3. Formal mismatch: RL optimises for reward maximisation. Curiosity is information gain for its own sake. These are different objectives.

### Why Global Workspace Theory

GWT (Baars 1988) provides a shared workspace to which specialised modules broadcast, and from which a unified signal is read out. It subsumes fast/slow weight distinctions and MANN-style memory, and adds a broadcast mechanism that enables personality structure rather than just memory retrieval.

Applied to TOPOS:
- The global workspace is a 384-dim EMA vector representing current discourse state.
- Specialised modules (episodic memory, concept graph, affective state) each write to this workspace.
- The SLM reads from the workspace to generate language. It is not the mind. It is the mouth.

GWT provides a formal connection to curiosity: inputs that do not fit the current workspace state produce a surprise signal, which triggers deeper processing. This is equivalent to information gain without a reward signal.

### Alternatives Considered

**Standard RL:** Rejected. See above.

**Liquid/Spiking Neural Networks:** Rejected. The interface problem between continuous embedding space and spike trains adds latency and information loss with no benefit for text. Revisit if TOPOS extends to audio or sensor streams.

**Dual-weight architecture:** Subsumed. The frozen SLM is the slow weight system. The workspace modules are the fast weight system.

**MANN-only (Neural Turing Machine lineage):** Subsumed. The episodic memory module is MANN-inspired. But MANN alone can recall what was said; it cannot develop what the agent cares about. The concept graph and affective state provide structure that MANN lacks.

---

## Final Architecture

### Base SLM

Qwen2.5-7B base (not instruct), 4-bit quantised via Ollama. Fully frozen during deployment. No gradients flow through it.

Base not instruct: instruct variants carry RLHF-baked personas and refusal behaviours that cannot be removed. The base model is a blank slate. All personality comes from the workspace.

### Workspace Modules

**Episodic Memory:** ChromaDB in-memory vector store. `write(text, metadata)` embeds and stores. `recall(query, n)` returns cosine-similar results with recency weighting. UUID-suffixed collection names prevent A/B instance collision.

**Concept Graph:** NetworkX DiGraph. Nodes are concept strings with accumulated surprise weights. Edges are directed co-occurrence pairs. Per-turn multiplicative decay (`GRAPH_DECAY=0.95`) prevents attractor lock-in over long runs. Pruned at `MAX_GRAPH_NODES=200`. Personality is graph topology.

**Affective State:** 64-dim NumPy vector. Updated via `state = 0.95*state + 0.3*stimulus`. Stimulus encodes surprise uniformly; first half is sign-flipped by sentiment to encode valence. Readouts: `arousal = norm(state)/sqrt(64)`, `valence = mean(state[:32])`.

**Surprise Signal:** `S_t = 1 - cosine(embedding_t, workspace_state_{t-1})`. High surprise drives larger graph weight increments, stronger affective updates, and salient episodic writes.

**Workspace State:** 384-dim EMA vector: `state = 0.6*state + 0.4*embedding`. Represents the centre of gravity of recent discourse. This is what surprise is measured against.

### Curiosity Mechanism

An ensemble of 5 MLPs predicts `z_{t+1}` from `(z_t, a_t)`. Epistemic uncertainty = mean per-dimension variance across ensemble predictions. When ensemble members disagree, the model is in unfamiliar state space. This is the exploration trigger.

Online training (measure-then-update) is required. Batch-trained ensembles produce monotonically rising epistemic as the workspace drifts beyond the training distribution. Online training keeps the ensemble current.

Init diversity is required. Scaling each member's weights by `[0.5, 0.8, 1.0, 1.5, 2.0]` at initialisation creates genuinely different hypotheses. Without this, epistemic collapses to O(10^-4) and carries no signal.

The AdaptiveThreshold gates autonomous exploration at the 85th percentile of recent epistemic values.

---

## Signal Quality: The Three-Run Story

Upstream signal quality dominates everything downstream. A noisy concept graph poisons all modules that depend on it.

**Run 1:** Concept extraction by word length (>4 chars) captured generic words ("about", "feels", "there's") while domain terms ("kernel", "improvisation") appeared once and accumulated no weight. Longitudinal experiment collapsed entirely to music. Systems was invisible despite 25 turns.

**Run 2:** Domain stopword list (~25 words) shifted top-10 concepts from 2/10 signal to 7/10 signal. Cross-domain synthesis appeared for the first time in Q2. Q3 still music-dominant. Valence permanently neutral: TextBlob returns near-zero polarity on technical and philosophical text, making the affect module non-functional.

**Run 3:** Three simultaneous fixes, all necessary:
1. Stopword list expanded (+25 domain-specific words)
2. TextBlob replaced with a 70-entry domain lexicon at 8x normalisation. Valence became non-zero.
3. Context string reframed from suggestive ("let it subtly shape your response") to directive first-person ("You find yourself drawn to: [concepts]. These are not topics, they are how you think.").

Results: Q1, Q2, Q3 all showed cross-domain synthesis. Valence resolved to negative, consistent with the frustrated/searching priming corpus. All three modules coherent for the first time.

---

## The Mind vs. Mouth Problem

Experiment 6 exposed a fundamental measurement problem: the SLM is an unreliable proxy for workspace state.

When the exploration prompt was changed from "generate 5 questions" to open-ended (R2), the SLM stopped producing questions entirely. All four gate-fired outputs were declarative paragraphs. The old QuestionGenerator was measuring prompt compliance, not genuine curiosity.

Additionally, R1 (feeding the SLM response back through the workspace) accelerated echo chamber convergence: systems-heavy SLM output reinforced systems concepts in the graph, which tightened the context string toward systems, which produced more systems output.

**Implication:** The correct measurement target is the workspace state itself, not SLM output. Graph topology, affect trajectory, and epistemic uncertainty curves are direct properties of the mind. SLM output is a lossy, biased translation.

---

## Known Weaknesses

1. **Concept extraction is heuristic.** The spaCy NER + POS pipeline is an improvement but still produces false positives. A learned extractor would be better.
2. **Base model closing conventions override workspace at response endings.** Workspace conditioning is strongest in response bodies. The prescribed fix is soft prompt projection (P-Tuning v2).
3. **Identity persistence under perturbation is unvalidated.** 50-turn accumulation works. Stability under direct contradiction is untested.
4. **No introspection.** The system is conditioned by its workspace state but cannot reason about it. True self-awareness would require the SLM to have read/write access to workspace as a first-class reasoning object.
5. **Affect saturates on long runs.** EMA decay of 0.95 is too slow for 500-turn priming. Arousal pegs at 1.0 and never recovers.
6. **Round-robin concept selection during autonomous turns** is deterministic. A curiosity-weighted selector would be more principled.
7. **ActionProjection is frozen and random.** The action vector conflates input novelty with agent choice.

---

## Experimental Results Summary

| Criterion | Exp 1 (A/B) | Exp 2 (Run 1) | Exp 3 (Run 2) | Exp 3 (Run 3) | Exp 5 | Exp 7 |
|---|---|---|---|---|---|---|
| Workspace divergence detectable | PASS | | | | | |
| Systems theme survives longitudinal | | FAIL | PASS | PASS | | |
| Cross-domain synthesis | | FAIL | Partial | PASS | | |
| Personality in Q3 | | FAIL | Partial | PASS | | |
| Epistemic tracks transitions | | | | | 1.05x PASS | 2.41x PASS |
| PE tracks transitions | | | | | 0.97x FAIL | 1.34x PASS |
| Progress signal valid | | | | | FAIL (short) | PASS (500t) |

---

## Open Question

Whether a system with a concept graph, affective state, and surprise-driven attention constitutes genuine curiosity or a functional equivalent remains philosophically open. The experiments show that the functional consequences are consistent with what we would expect from a genuinely individualised agent. Whether functional equivalence constitutes the real thing is a question the architecture cannot answer about itself.
