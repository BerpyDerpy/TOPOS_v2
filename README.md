# TOPOS

TOPOS is a Global Workspace Theory inspired cognitive architecture for conversational agents. Instead of relying on reinforcement learning or a prompted persona, it separates cognition from language generation: a small set of workspace modules (a concept graph, an affective state vector, episodic memory, and a surprise signal) hold and update the agent's state, and a frozen small language model reads from that workspace purely to produce text. The model doesn't think here. It speaks what the workspace already holds.

Development started April 18, 2026, and the bulk of the architecture and experiments described in RESEARCH.md were done through early May.

## What's in here

- **Curiosity without RL.** An ensemble of forward models predicts the next workspace state, and disagreement across the ensemble (epistemic uncertainty) is used as the signal for exploration. Across our longitudinal experiments this tracked domain transitions in conversation better than raw prediction error (2.41x vs 1.34x transition ratio in the final run).
- **Personality as graph structure.** Personality is treated as an emergent property of a decaying, weighted concept graph built from conversational surprise, rather than a system prompt. Getting this working cleanly took a few iterations, mainly around concept extraction and sentiment, documented in RESEARCH.md.
- **Output as a lossy readout of state.** One of the more useful things we found along the way: the language model's output turned out to be an unreliable proxy for what the workspace was actually doing internally. Changing the exploration prompt changed the model's surface behavior sharply while the underlying workspace state hadn't moved, and feeding model output back into the workspace directly caused echo chamber effects. We ended up treating workspace state (graph topology, affect trajectory, epistemic uncertainty) as the thing worth measuring, not the model's text.
- **Signal quality dominates everything else.** The clearest lesson from the three iterated runs: a noisy concept extraction step quietly wrecks every downstream module, regardless of how well designed the architecture is on paper.

Full experimental writeup, including the failure modes and what didn't work, is in RESEARCH.md.

This is ongoing, imperfect work with plenty of open questions (identity persistence under contradiction, affect saturation over long runs, no real introspection). Feedback and issues welcome.
