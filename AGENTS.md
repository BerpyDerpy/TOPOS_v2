# AGENTS.md - TOPOS Design Reference

**TOPOS** (Topology Oriented Persistent Observer System) is an experiment in emergent curiosity and self-learning without reinforcement learning. The core hypothesis: a system can develop genuine exploratory behaviour through internal state dynamics alone, with no external reward signal.

---

## What We Are Building

A conversational agent that accumulates a persistent internal state across turns. That state shapes how it responds, what it attends to, and when it initiates exploration. Personality and curiosity live in the workspace, not in the language model's weights.

Three sub-problems:
- **Continuous learning without catastrophic forgetting.** All learning is encoded in workspace state. The SLM is frozen.
- **Personality without programming.** Character emerges from the topology of the concept graph, not from a system prompt.
- **Curiosity without prompt.** The agent initiates exploration when its forward model is uncertain, not when instructed to.

The SLM is the mouth. The workspace is the mind.

---

## Architecture

TOPOS implements Global Workspace Theory (Baars 1988). A shared workspace vector is updated each turn by specialised modules. The SLM reads from that workspace to generate language.

### Modules

| File | Role | State |
|---|---|---|
| `config.py` | All constants | Stateless |
| `embedder.py` | MiniLM-L6-v2 wrapper, `embed()` + `similarity()` | Frozen model weights |
| `memory.py` | ChromaDB episodic store, `write()` + `recall()` | In-memory vector store |
| `affect.py` | 64-dim affective vector, arousal + valence readouts | NumPy array |
| `graph.py` | Weighted DiGraph of concepts, decays per turn | NetworkX DiGraph |
| `nlp_extractor.py` | spaCy NER + POS + noun chunk pipeline, returns `WeightedConcept` list | Lazy-loaded spaCy model |
| `workspace.py` | Orchestrator: runs the cognitive loop, builds context string, calls SLM | 384-dim EMA state vector |
| `state_encoder.py` | Encodes all four modules into z_t in R^448 | Four frozen orthogonal projections |
| `forward_model.py` | 5-member MLP ensemble predicting z_{t+1} from (z_t, a_t), epistemic uncertainty | Trainable MLP weights + Adam state |
| `integration.py` | Wraps workspace with curiosity tracking, autonomous exploration, JSONL logging | Orchestration layer |
| `main.py` | CLI entry point for all experiment modes | None |

### Single-Turn Data Flow

```
input text
  -> embed (384-dim)
  -> surprise = 1 - cosine(embedding, workspace_state)
  -> EMA update workspace state
  -> extract concepts (spaCy pipeline)
  -> graph.decay() then graph.update(concepts, surprise)
  -> affect.update(surprise, lexicon_sentiment)
  -> memory.write(text, turn)
  -> context_string() assembled from graph + affect + memory recall
  -> SLM generates response from context_string + input
```

### State Encoder (z_t in R^448)

```
z_t = [z_graph | z_affect | z_memory | z_input]
       R^128     R^64       R^128      R^128
```

Each subvector is a frozen orthogonal linear projection. These are never trained. The forward model learns on top of them.

- `z_graph`: weight-normalised mean of top-10 concept embeddings
- `z_affect`: direct projection of the 64-dim affect vector
- `z_memory`: recency x similarity soft attention over all stored episode embeddings (half-life = 10 turns)
- `z_input`: direct projection of the current workspace state vector

### Forward Model and Curiosity Signal

An ensemble of 5 MLPs, each `Linear(576->256) -> GELU -> Linear(256->256) -> GELU -> Linear(256->448)`. Input is `concat(z_t, a_t)` where `a_t` is a frozen orthogonal projection of the input embedding into R^128.

**Epistemic uncertainty** = mean per-dimension variance across ensemble predictions before seeing ground truth. This is the exploration signal. High epistemic = ensemble members disagree = state space is unfamiliar.

**Three signals per turn:**
- `epistemic`: ensemble disagreement (before update). Used to gate exploration.
- `error`: L2 norm of (mean_pred - z_t1_actual). Diagnostic only.
- `progress`: rolling_mean_error - current_error (K=10 window). Positive = model is learning.

**AdaptiveThreshold**: 85th percentile of recent epistemic values. Inputs above it trigger autonomous exploration.

**Critical implementation details validated by experiments:**
- Online training is required (measure-then-update). Batch training produces monotonically drifting epistemic, not domain signal.
- Init diversity is required: weight scales `[0.5, 0.8, 1.0, 1.5, 2.0]` applied to each member at startup. Without this, epistemic collapses to O(10^-4).
- Bagging keep_prob=0.6 maintains ensemble disagreement during training.

### Autonomous Exploration (integration.py)

When no user input arrives:
1. Pull top-10 concepts from graph, select one by `turn_index % len(concepts)` (round-robin)
2. Recall one memory using that concept as query
3. Process the recalled memory through workspace (no SLM)
4. Measure epistemic uncertainty on the resulting state transition
5. If `should_explore()`: generate free-form text via SLM from current context string, feed response back through workspace (R1 learning loop closure)
6. If gate closed: train forward model on recall transition and idle

The exploration prompt does NOT instruct the SLM to ask questions. Whether it does is logged as `has_questions` and treated as an observation, not a command (R2).

---

## Key Config Values

```python
WORKSPACE_ALPHA  = 0.4    # EMA blend rate: state = 0.6*state + 0.4*embedding
AFFECT_ALPHA     = 0.3    # affect EMA weight
AFFECT_DECAY     = 0.95   # per-turn affect decay toward zero
GRAPH_DECAY      = 0.95   # per-turn multiplicative decay on graph weights (was 0.99, caused attractor lock-in)
MAX_GRAPH_NODES  = 200
SENTIMENT_SCALE  = 8.0
WORKSPACE_DIM    = 384    # must match embed model output
AFFECT_DIM       = 64
# Z_DIM = 448 (from state_encoder.py)
```

---

## Known Architectural Flags

These are issues worth being aware of when making changes:

- **Round-robin concept selection** during autonomous turns (`turn_index % len(concepts)`) is deterministic, not drive-based. A curiosity-weighted selector would be more principled.
- **Affect is read-only from the behavioral layer.** Arousal and valence appear in the context string but do not influence the exploration gate or concept selection. This is a gap.
- **ActionProjection is frozen.** The action vector `a_t` is a random orthogonal projection of the input embedding, not a learned representation of what the agent chose to do.
- **Sentiment lexicon is manually curated** for music and systems domains. Emotional salience is not emergent.
- **Init diversity decays silently.** Ensemble weight norms converge over long runs (Exp 7 H4 FAIL). No mechanism restores diversity once lost.
- **SURPRISE_THRESHOLD = 0.35** is defined in config but unused. Reserved for arousal-gated memory writes.
- **STOPWORDS set in workspace.py** is dead code since nlp_extractor.py replaced the old approach.

---

## Running Experiments

```bash
python main.py --chat                         # interactive REPL
python main.py --experiment                   # A/B workspace divergence
python main.py --longitudinal                 # 50-turn interleaved priming
python main.py --autonomous                   # 50-turn priming + autonomous phase
python main.py --mega-longitudinal            # 500-turn/domain corpus experiment
python main.py --experiment-6                 # 250+250 corpus priming + autonomous
```

Corpus files expected at `corpus/music_500.jsonl` and `corpus/systems_500.jsonl`. All experiment turns log to `runs/`.

---

## The Open Question

Whether epistemic uncertainty + surprise-driven workspace evolution constitutes genuine curiosity or a functional equivalent is unresolved. What the experiments show is that the workspace state genuinely evolves with input, that domain transitions produce measurable epistemic spikes (2.41x ratio, Exp 7), and that the SLM output is an unreliable proxy for measuring that internal state. The correct measurement target is the workspace itself: graph topology, affect trajectory, epistemic curves.
