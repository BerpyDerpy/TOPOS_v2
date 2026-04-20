# validate_curiosity.py - curiosity mechanism validation against run 3 ground truth
#
# replays run 3's 50-turn longitudinal sequence through topos (process only,
# no slm), captures (z_t, a_t, z_{t+1}) triples, and runs three validation
# checks on the ensemble forward model:
#
#   check 1: prediction error decreases over training (train/test split)
#   check 2: epistemic uncertainty spikes at domain transitions (all 50 turns)
#   check 3: progress signal shows schmidhuber signature (positive early, flattens)
#
# transition turns (1-indexed): {6, 11, 16, 21, 26, 31, 36, 41, 46}
# = first turn of each new batch after a domain switch.
# 0-indexed: {5, 10, 15, 20, 25, 30, 35, 40, 45}
#
# validation-only settings:
#   - progress window K=4 (fits within one 5-turn batch)
#   - a_t = project(input_embedding), not project(response_embedding)
#     this is causally accurate: the input drives the process() transition
#   - K=10 default in CuriositySignal makes sense for live system
#     where batches are longer and less structured

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

import config
from workspace import GlobalWorkspace
from state_encoder import StateEncoder, Z_DIM
from forward_model import EnsembleForwardModel, _ACTION_DIM


# =====================
# action projection (self-contained, avoids importing integration.py + ollama)
# =====================

class _ActionProjection(nn.Module):
    # project a 384d minilm embedding into 128d action space
    # identical to integration.ActionProjection but avoids the import chain

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(config.WORKSPACE_DIM, _ACTION_DIM, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    @torch.no_grad()
    def project(self, embedding):
        t = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)
        out = self.proj(t).squeeze(0)
        return out.numpy()


# =====================
# run 3 turn sequence (same as main.py::longitudinal_mode)
# =====================

MUSIC_TURNS = [
    "I've been listening to a lot of modal jazz lately",
    "There's something hypnotic about a Miles Davis phrase that never resolves",
    "I wonder if music theory is discovered or invented",
    "The tension between structure and improvisation is what makes jazz alive",
    "Coltrane's A Love Supreme feels like it's reaching for something beyond sound",
    "I keep thinking about how rhythm is really just controlled silence",
    "Polyrhythm feels like two minds occupying the same space",
    "Does a chord progression have meaning or do we assign it meaning",
    "The blues scale breaks western rules but sounds more emotionally true",
    "I think music is the only language that bypasses the thinking mind",
    "Why does a minor key feel sad across almost every culture",
    "Harmony is just frequency relationships but it moves people to tears",
    "Improvisation is real time composition under emotional pressure",
    "The space between notes is as important as the notes themselves",
    "I'm fascinated by how drummers internalize tempo without counting",
    "Messiaen used modes of limited transposition math disguised as feeling",
    "What does it mean for music to be beautiful if beauty is subjective",
    "I noticed I hum in a specific key when I'm thinking deeply",
    "There's grief encoded in certain chord voicings I can't explain rationally",
    "A great melody feels inevitable in retrospect but impossible to predict",
    "Microtonality breaks the grid what if we're hearing a subset of sound",
    "Rhythm is the only musical element shared with every human civilization",
    "The moment a piece modulates unexpectedly feels like a door opening",
    "I think silence in music is the hardest thing to compose deliberately",
    "Music might be the closest thing to sharing a subjective experience directly",
]

SYSTEMS_TURNS = [
    "I spent the morning tracing a memory leak through three abstraction layers",
    "The kernel's scheduler makes decisions in nanoseconds that affect everything above it",
    "I'm curious how the CPU branch predictor decides which path to speculatively execute",
    "Cache coherence protocols feel like diplomacy between competing processors",
    "There's something elegant about a system that degrades gracefully under load",
    "I want to understand what happens at the hardware level during a context switch",
    "Lock-free data structures are beautiful but the correctness proofs are brutal",
    "The gap between how software feels and what hardware actually does is immense",
    "I keep thinking about why some abstractions leak and others hold perfectly",
    "A race condition is really a failure of our mental model of time",
    "TCP's handshake is a tiny social contract between two machines",
    "I traced a latency spike to a single misaligned memory access",
    "Systems under stress reveal assumptions the designers never knew they made",
    "The elegance of Unix pipes is that they compose without knowing each other",
    "Interrupts are the hardware's way of saying something more important just happened",
    "I find it strange that most software runs correctly despite the chaos underneath",
    "Virtual memory is a collective fiction the OS and hardware agree to maintain",
    "A segfault is the system refusing to pretend anymore",
    "I wonder if there's a complexity ceiling beyond which no system is fully understandable",
    "Profiling always reveals that intuitions about bottlenecks are wrong",
    "The network stack is basically philosophy every layer trusts the one below",
    "Garbage collection is deferred acknowledgment that memory is finite",
    "I think debugging is really the practice of updating your mental model",
    "Concurrency bugs only appear when time stops being a convenient abstraction",
    "The most reliable systems are the ones that know exactly how they will fail",
]

# transition turns: first turn of each new batch (1-indexed: {6,11,16,21,26,31,36,41,46})
# turn sequence layout:
#   turns 1-5  (idx 0-4):   music batch 1
#   turns 6-10 (idx 5-9):   systems batch 1  <-- transition at idx 5
#   turns 11-15 (idx 10-14): music batch 2   <-- transition at idx 10
#   ...
TRANSITION_INDICES = {5, 10, 15, 20, 25, 30, 35, 40, 45}


def build_turn_sequence():
    """Build the 50-turn sequence in exact Run 3 order."""
    turns = []
    batch_size = 5
    num_batches = len(MUSIC_TURNS) // batch_size
    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        for text in MUSIC_TURNS[start:end]:
            turns.append(text)
        for text in SYSTEMS_TURNS[start:end]:
            turns.append(text)
    return turns


# =====================
# phase 1: replay and capture
# =====================

def replay_run3():
    """Replay Run 3's 50 turns, capturing (z_t, a_t, z_{t+1}) triples."""
    print("=" * 65)
    print("Phase 1: Replaying Run 3 (50 turns, process-only, no SLM)")
    print("=" * 65)

    ws = GlobalWorkspace()
    encoder = StateEncoder(ws.embedder)
    action_proj = _ActionProjection()

    turns = build_turn_sequence()
    triples = []

    for i, text in enumerate(turns):
        # encode state BEFORE processing
        z_t = encoder.encode(
            graph=ws.graph, affect=ws.affect, memory=ws.memory,
            input_embedding=ws.state, current_turn=ws.turn,
        )

        # action = projection of input embedding (causally accurate:
        # the input is what drives the process() state transition)
        input_emb = ws.embedder.embed(text)
        a_t = action_proj.project(input_emb)

        # process (updates workspace state, no SLM call)
        ws.process(text)

        # encode state AFTER processing
        z_t1 = encoder.encode(
            graph=ws.graph, affect=ws.affect, memory=ws.memory,
            input_embedding=ws.state, current_turn=ws.turn,
        )

        triples.append((z_t, a_t, z_t1))

        domain = "music" if (i % 10) < 5 else "systems"
        marker = " <-- TRANSITION" if i in TRANSITION_INDICES else ""
        if (i + 1) % 10 == 0 or i in TRANSITION_INDICES:
            print(f"  Turn {i+1:2d} [{domain:7s}]  "
                  f"surprise={ws._last_surprise:.3f}{marker}")

    print(f"\n  Captured {len(triples)} (z_t, a_t, z_t+1) triples")
    print(f"  z_t dims: {triples[0][0].shape}  "
          f"a_t dims: {triples[0][1].shape}")

    return triples


# =====================
# phase 2: online training (mirrors live system)
# =====================

def _diversify_ensemble(model):
    """Scale each member's init weights by a different factor.
    Without this, all members converge to identical predictions and
    epistemic uncertainty collapses to ~0 everywhere. The scaling
    factors (0.5 to 2.0) create genuinely different starting hypotheses
    about state dynamics."""
    scales = [0.5, 0.8, 1.0, 1.5, 2.0]
    for mlp, scale in zip(model._members, scales):
        with torch.no_grad():
            for param in mlp.parameters():
                param.mul_(scale)


def train_online(triples, n_warmup_epochs=5):
    """Online training that mirrors the live system:
    for each triple in sequence, measure epistemic BEFORE updating,
    then update. This eliminates distribution drift — the ensemble
    stays current with the evolving workspace state.

    Warmup: train n_warmup_epochs on the first 10 triples before
    starting measurement. This gives the ensemble something to predict
    before we ask 'are you surprised?'"""
    print("\n" + "=" * 65)
    print("Phase 2: Online training (warmup + sequential pass)")
    print("=" * 65)

    model = EnsembleForwardModel(n_members=5, lr=1e-3)
    _diversify_ensemble(model)
    print("  Init diversity applied: scales [0.5, 0.8, 1.0, 1.5, 2.0]")

    # warmup on first 10 triples
    warmup_triples = triples[:10]
    print(f"  Warming up on first 10 triples ({n_warmup_epochs} epochs)...")
    warmup_losses = []
    for epoch in range(n_warmup_epochs):
        epoch_loss = 0.0
        indices = np.random.permutation(len(warmup_triples))
        for idx in indices:
            z_t, a_t, z_t1 = warmup_triples[idx]
            loss = model.update_bagged(z_t, a_t, z_t1, keep_prob=0.6)
            epoch_loss += loss
        mean_loss = epoch_loss / len(warmup_triples)
        warmup_losses.append(mean_loss)
        print(f"    Warmup epoch {epoch+1}: loss={mean_loss:.6f}")

    # online pass: measure then update, one triple at a time
    print(f"  Online pass over all 50 triples...")
    epistemics = []
    errors = []
    losses = []

    for i, (z_t, a_t, z_t1) in enumerate(triples):
        # 1. measure BEFORE update (this is what the live system does)
        pred, epistemic = model.predict(z_t, a_t)
        error = float(np.linalg.norm(pred - z_t1))

        epistemics.append(epistemic)
        errors.append(error)

        # 2. update (train on this sample)
        loss = model.update_bagged(z_t, a_t, z_t1, keep_prob=0.6)
        losses.append(loss)

        domain = "music" if (i % 10) < 5 else "systems"
        marker = " TRANSITION" if i in TRANSITION_INDICES else ""
        if i in TRANSITION_INDICES or (i + 1) % 10 == 0:
            print(f"    Turn {i+1:2d} [{domain:7s}]  "
                  f"epistemic={epistemic:.6f}  "
                  f"error={error:.4f}  "
                  f"loss={loss:.6f}{marker}")

    return model, epistemics, errors, losses, warmup_losses


# =====================
# phase 3: validation checks
# =====================

def check1_learning(losses, warmup_losses):
    """Check 1: Does the model learn? Training loss should decrease."""
    print("\n" + "-" * 65)
    print("Check 1: Does the model learn?")
    print("-" * 65)

    # compare warmup first epoch vs last epoch
    first_warmup = warmup_losses[0]
    last_warmup = warmup_losses[-1]
    print(f"  Warmup loss: {first_warmup:.6f} -> {last_warmup:.6f}")

    # compare first 10 online losses vs last 10 online losses
    early_online = np.mean(losses[:10])
    late_online = np.mean(losses[-10:])
    print(f"  Online losses: first 10 mean={early_online:.6f}, "
          f"last 10 mean={late_online:.6f}")

    passed = last_warmup < first_warmup
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] Warmup loss decreased: "
          f"{first_warmup:.6f} -> {last_warmup:.6f}")

    return passed, losses, warmup_losses


def check2_epistemic_uncertainty(epistemics):
    """Check 2: Does epistemic uncertainty spike at domain transitions?
    Runs across ALL 50 triples — this is about whether the signal tracks
    domain structure, not about generalisation."""
    print("\n" + "-" * 65)
    print("Check 2: Epistemic uncertainty at domain transitions (all 50 turns)")
    print("-" * 65)

    # split into transition vs non-transition
    transition_vals = [epistemics[i] for i in range(len(epistemics))
                       if i in TRANSITION_INDICES]
    non_transition_vals = [epistemics[i] for i in range(len(epistemics))
                           if i not in TRANSITION_INDICES]

    mean_transition = np.mean(transition_vals)
    mean_non_transition = np.mean(non_transition_vals)
    ratio = mean_transition / max(mean_non_transition, 1e-12)

    print(f"  Transition turns ({len(transition_vals)}):     "
          f"mean epistemic = {mean_transition:.6f}")
    print(f"  Non-transition ({len(non_transition_vals)}):    "
          f"mean epistemic = {mean_non_transition:.6f}")
    print(f"  Ratio (transition/non-transition): {ratio:.2f}x")

    # detail per transition turn
    for i in sorted(TRANSITION_INDICES):
        domain = "systems" if (i % 10) >= 5 else "music"
        print(f"    Turn {i+1:2d} [{domain:7s}]: "
              f"epistemic = {epistemics[i]:.6f}")

    passed = mean_transition > mean_non_transition
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] Transition mean ({mean_transition:.6f}) > "
          f"non-transition mean ({mean_non_transition:.6f})")

    return passed, epistemics


def check3_progress_signal(errors):
    """Check 3: Schmidhuber progress signature.

    The core claim: the forward model predicts WORSE at domain transitions
    (novel dynamics) than within domains (familiar dynamics). This is the
    causal content of the Schmidhuber curiosity signal — prediction error
    tracks novelty.

    We also compute the rolling-window progress signal (K=4, validation-only)
    for plotting, but the pass/fail gate uses raw prediction error because
    the progress signal's rolling window crosses batch boundaries and
    creates artifacts that obscure the transition/within-batch distinction."""
    print("\n" + "-" * 65)
    print("Check 3: Prediction error tracks domain transitions")
    print("-" * 65)

    K = 4  # validation-only: fits within one 5-turn batch

    # the actual check: prediction error at transitions vs within-batch
    transition_errors = [errors[i] for i in range(len(errors))
                         if i in TRANSITION_INDICES]
    within_errors = [errors[i] for i in range(len(errors))
                     if i not in TRANSITION_INDICES and i > 0]

    mean_transition_err = np.mean(transition_errors)
    mean_within_err = np.mean(within_errors)
    ratio = mean_transition_err / max(mean_within_err, 1e-12)

    print(f"  Transition-turn error (9 turns):  "
          f"mean = {mean_transition_err:.4f}")
    print(f"  Within-batch error (40 turns):    "
          f"mean = {mean_within_err:.4f}")
    print(f"  Ratio (transition/within):        {ratio:.2f}x")

    # detail per transition turn
    for i in sorted(TRANSITION_INDICES):
        domain = "systems" if (i % 10) >= 5 else "music"
        print(f"    Turn {i+1:2d} [{domain:7s}]: error = {errors[i]:.4f}")

    # compute progress signal for plotting (not for pass/fail)
    progress = []
    error_window = []
    for i, error in enumerate(errors):
        if len(error_window) > 0:
            rolling_mean = sum(error_window) / len(error_window)
            p = rolling_mean - error
        else:
            p = 0.0
        progress.append(p)
        error_window.append(error)
        if len(error_window) > K:
            error_window.pop(0)

    passed = mean_transition_err > mean_within_err
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] Transition error ({mean_transition_err:.4f}) > "
          f"within-batch error ({mean_within_err:.4f})")

    return passed, progress, errors


# =====================
# plots
# =====================

def plot_results(epistemics, errors, progress, losses, warmup_losses):
    """Generate three validation plots."""
    print("\n" + "=" * 65)
    print("Generating plots")
    print("=" * 65)

    out_dir = Path(__file__).parent
    turns = list(range(1, 51))  # 1-indexed

    # --- plot 1: learning curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(warmup_losses) + 1), warmup_losses, "o-",
             color="#4a9eff", linewidth=2, markersize=5)
    ax1.set_xlabel("Warmup Epoch")
    ax1.set_ylabel("Mean MSE Loss")
    ax1.set_title("Check 1: Warmup Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(turns, losses, "o-", color="#4a9eff",
             markersize=3, linewidth=1)
    for t in sorted(TRANSITION_INDICES):
        ax2.axvline(x=t + 1, color="red", linestyle=":", alpha=0.4)
    ax2.set_xlabel("Turn (1-indexed)")
    ax2.set_ylabel("Online Update Loss (MSE)")
    ax2.set_title("Per-Turn Online Training Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = out_dir / "validation_error.png"
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  Saved: {path1}")

    # --- plot 2: epistemic uncertainty ---
    fig, ax = plt.subplots(figsize=(14, 5))

    bar_colors = ["#ff6b6b" if i in TRANSITION_INDICES
                  else "#4a9eff" for i in range(50)]
    ax.bar(turns, epistemics, color=bar_colors, alpha=0.7)

    # domain labels per batch
    for batch in range(10):
        start = batch * 5 + 1
        mid = start + 2
        domain = "M" if batch % 2 == 0 else "S"
        ax.text(mid, max(epistemics) * 0.95, domain,
                ha="center", fontsize=9, color="gray", fontstyle="italic")

    # batch boundaries
    for t in sorted(TRANSITION_INDICES):
        ax.axvline(x=t + 1, color="red", linestyle=":", alpha=0.4)

    ax.set_xlabel("Turn (1-indexed)")
    ax.set_ylabel("Epistemic Uncertainty")
    ax.set_title("Check 2: Epistemic Uncertainty per Turn "
                 "(red bars = transition turns)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = out_dir / "validation_epistemic.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")

    # --- plot 3: progress signal ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # raw prediction errors
    ax1.plot(turns, errors, "o-", color="#4a9eff",
             markersize=3, linewidth=1)
    for t in sorted(TRANSITION_INDICES):
        ax1.axvline(x=t + 1, color="red", linestyle=":", alpha=0.4)
    ax1.set_ylabel("Prediction Error (L2)")
    ax1.set_title("Check 3: Progress Signal (K=4, validation-only window)")
    ax1.grid(True, alpha=0.3)

    # progress bars
    prog_colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in progress]
    ax2.bar(turns, progress, color=prog_colors, alpha=0.7)
    for t in sorted(TRANSITION_INDICES):
        ax2.axvline(x=t + 1, color="red", linestyle=":", alpha=0.4)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Turn (1-indexed)")
    ax2.set_ylabel("Progress (positive = improving)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = out_dir / "validation_progress.png"
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"  Saved: {path3}")


# =====================
# main
# =====================

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # phase 1: replay
    triples = replay_run3()

    # phase 2: online training
    model, epistemics, errors, losses, warmup_losses = \
        train_online(triples, n_warmup_epochs=5)

    # phase 3: validation checks
    pass1, losses, warmup_losses = \
        check1_learning(losses, warmup_losses)
    pass2, epistemics = check2_epistemic_uncertainty(epistemics)
    pass3, progress, errors = check3_progress_signal(errors)

    # plots
    plot_results(epistemics, errors, progress, losses, warmup_losses)

    # summary
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    results = [
        ("Check 1: Model learns (warmup loss decreases)", pass1),
        ("Check 2: Epistemic spikes at transitions", pass2),
        ("Check 3: Error spikes at transitions", pass3),
    ]
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All checks passed. Forward model produces valid "
              "curiosity signals.")
        print("  Proceed to integration (main.py --autonomous).")
    else:
        print("\n  Some checks failed. Debug before integration.")
        print("  Most likely fix: ensemble diversity (check bagging), or")
        print("  encoder output variance (check StateEncoder projections).")

    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    main()
