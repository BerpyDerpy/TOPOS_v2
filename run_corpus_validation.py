# run_corpus_validation.py - 500-turn corpus validation run
#
# standalone script (not wired into main.py). loads the music corpus via
# corpus_loader.py, replays each turn through GlobalWorkspace.process()
# (no SLM), runs the forward model online with experiment 5 settings,
# and logs all signals to runs/music_validation_<timestamp>.jsonl
#
# what this checks (single-domain, 10-stage arc):
#   - does prediction error form a declining baseline within stages?
#   - do top concepts shift meaningfully between stage 1 and stage 10?
#   - do ensemble weight norms diverge or converge over time?
#   - does the progress signal produce non-zero readings by stage 3+?
#
# what this does NOT check:
#   - cross-domain epistemic spikes (this is single-domain music only)

from __future__ import annotations

import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from workspace import GlobalWorkspace
from state_encoder import StateEncoder, Z_DIM
from forward_model import EnsembleForwardModel, CuriositySignal, _ACTION_DIM
from corpus_loader import CorpusLoader, CorpusLoadError


# =====================
# action projection (self-contained, avoids importing integration.py + ollama)
# =====================

class _ActionProjection(nn.Module):
    """Project 384d MiniLM embedding into 128d action space.
    Identical to integration.ActionProjection but avoids the import chain
    that pulls in ollama."""

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
# ensemble setup (experiment 5 settings)
# =====================

ENSEMBLE_MEMBERS = 5
ENSEMBLE_LR      = 1e-3
BAGGING_KEEP     = 0.6
INIT_SCALES      = [0.5, 0.8, 1.0, 1.5, 2.0]
PROGRESS_WINDOW  = 10


def build_ensemble():
    """Build forward model with experiment 5 init diversity."""
    model = EnsembleForwardModel(n_members=ENSEMBLE_MEMBERS, lr=ENSEMBLE_LR)

    for mlp, scale in zip(model._members, INIT_SCALES):
        with torch.no_grad():
            for param in mlp.parameters():
                param.mul_(scale)

    return model


def get_ensemble_weight_norms(model):
    """Return list of L2 norms of flattened weight vectors per member."""
    norms = []
    for mlp in model._members:
        total = 0.0
        for param in mlp.parameters():
            total += param.data.norm().item() ** 2
        norms.append(total ** 0.5)
    return norms


# =====================
# main run
# =====================

def run_validation():
    np.random.seed(42)
    torch.manual_seed(42)

    # ---- load corpus ----
    corpus_pattern = "corpus/systems_stage_*.jsonl"
    print("=" * 70)
    print("SYSTEMS CORPUS VALIDATION RUN")
    print("=" * 70)
    print(f"  Corpus: {corpus_pattern}")

    try:
        loader = CorpusLoader(corpus_pattern)
    except CorpusLoadError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"  Domain: {loader.domain}")
    print(f"  Total turns: {loader.total_turns}")
    print(f"  Total stages: {loader.total_stages}")

    # ---- build workspace + forward model ----
    ws = GlobalWorkspace()
    encoder = StateEncoder(ws.embedder)
    action_proj = _ActionProjection()
    model = build_ensemble()

    print(f"\n  Ensemble: {ENSEMBLE_MEMBERS} members, lr={ENSEMBLE_LR}, "
          f"bagging_keep={BAGGING_KEEP}")
    print(f"  Init scales: {INIT_SCALES}")
    print(f"  Progress window K={PROGRESS_WINDOW}")

    # ---- prepare log ----
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"system_validation_{timestamp}.jsonl"

    # ---- per-turn tracking ----
    all_records = []
    error_history = []  # for progress signal (rolling window)

    print(f"\n  Log: {log_path}")
    print()
    print("-" * 70)
    print("Running 500 turns (process-only, no SLM)...")
    print("-" * 70)

    current_stage = 0

    for global_turn, stage, text in loader:

        # stage boundary marker
        if stage != current_stage:
            current_stage = stage
            print(f"\n  === Stage {stage} ===")

        # encode state BEFORE processing
        z_t = encoder.encode(
            graph=ws.graph, affect=ws.affect, memory=ws.memory,
            input_embedding=ws.state, current_turn=ws.turn,
        )

        # action = projection of input embedding (causally accurate)
        input_emb = ws.embedder.embed(text)
        a_t = action_proj.project(input_emb)

        # 1. MEASURE: predict before seeing ground truth
        mean_pred, epistemic = model.predict(z_t, a_t)

        # process (updates workspace state, no SLM)
        ws.process(text)

        # encode state AFTER processing
        z_t1 = encoder.encode(
            graph=ws.graph, affect=ws.affect, memory=ws.memory,
            input_embedding=ws.state, current_turn=ws.turn,
        )

        # prediction error
        prediction_error = float(np.linalg.norm(mean_pred - z_t1))

        # progress signal
        if len(error_history) > 0:
            rolling_mean = sum(error_history[-PROGRESS_WINDOW:]) / min(
                len(error_history), PROGRESS_WINDOW
            )
            progress_signal = rolling_mean - prediction_error
        else:
            progress_signal = 0.0
        error_history.append(prediction_error)

        # 2. UPDATE: train on this sample
        model.update_bagged(z_t, a_t, z_t1, keep_prob=BAGGING_KEEP)

        # workspace readouts
        top_concepts = ws.graph.top_concepts(5)
        arousal = ws.affect.arousal()
        valence = ws.affect.valence()
        state_norm = float(np.linalg.norm(ws.state))
        weight_norms = get_ensemble_weight_norms(model)

        # batch_index = turn within current stage (1-50)
        batch_index = global_turn - (stage - 1) * 50

        record = {
            "global_turn_index": global_turn,
            "stage": stage,
            "batch_index": batch_index,
            "epistemic_uncertainty": epistemic,
            "prediction_error": prediction_error,
            "progress_signal": progress_signal,
            "top_5_concepts": top_concepts,
            "arousal": arousal,
            "valence": valence,
            "workspace_state_norm": state_norm,
            "ensemble_weight_norms": weight_norms,
        }
        all_records.append(record)

        # write to log immediately
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # periodic stdout
        if batch_index in {1, 25, 50}:
            print(f"    turn {global_turn:3d} (stage {stage}, batch {batch_index:2d})  "
                  f"epistemic={epistemic:.6f}  error={prediction_error:.4f}  "
                  f"progress={progress_signal:+.4f}")

    # ---- summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # group records by stage
    by_stage = {}
    for r in all_records:
        s = r["stage"]
        if s not in by_stage:
            by_stage[s] = []
        by_stage[s].append(r)

    # 1. mean epistemic per stage
    print("\n  Mean epistemic uncertainty per stage:")
    print(f"  {'Stage':>6}  {'Mean Epistemic':>16}  {'Mean Error':>12}")
    print(f"  {'-----':>6}  {'-------------':>16}  {'----------':>12}")
    for s in range(1, 11):
        records = by_stage[s]
        mean_ep = np.mean([r["epistemic_uncertainty"] for r in records])
        mean_err = np.mean([r["prediction_error"] for r in records])
        print(f"  {s:6d}  {mean_ep:16.6f}  {mean_err:12.4f}")

    # 2. stage boundary transition ratios
    # boundaries at global turns 50, 100, ..., 450 (last turn of each stage)
    # compare boundary epistemic vs mean of surrounding 10 turns
    print("\n  Stage boundary transition ratios:")
    print(f"  {'Boundary':>10}  {'Boundary Ep.':>14}  {'Surround Mean':>15}  {'Ratio':>8}")
    print(f"  {'--------':>10}  {'------------':>14}  {'-------------':>15}  {'-----':>8}")

    boundary_turns = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    record_by_turn = {r["global_turn_index"]: r for r in all_records}

    for bt in boundary_turns:
        boundary_ep = record_by_turn[bt]["epistemic_uncertainty"]

        # surrounding 10 turns: 5 before and 5 after the boundary
        # (the boundary itself is excluded)
        surround_turns = []
        for offset in range(-5, 6):
            t = bt + offset
            if t != bt and t in record_by_turn:
                surround_turns.append(t)

        surround_eps = [record_by_turn[t]["epistemic_uncertainty"]
                        for t in surround_turns]
        surround_mean = np.mean(surround_eps) if surround_eps else 0.0
        ratio = boundary_ep / max(surround_mean, 1e-12)

        print(f"  {bt:10d}  {boundary_ep:14.6f}  {surround_mean:15.6f}  {ratio:8.2f}x")

    # 3. ensemble weight norm divergence
    print("\n  Ensemble weight norm divergence (std across 5 members):")
    check_turns = [1, 100, 250, 500]
    print(f"  {'Turn':>6}  {'Norms':>50}  {'Std':>10}")
    print(f"  {'----':>6}  {'-----':>50}  {'---':>10}")

    for t in check_turns:
        if t in record_by_turn:
            norms = record_by_turn[t]["ensemble_weight_norms"]
            norms_str = "  ".join(f"{n:.3f}" for n in norms)
            std = np.std(norms)
            print(f"  {t:6d}  {norms_str:>50}  {std:10.4f}")

    # 4. top 5 concepts at key turns
    print("\n  Top 5 concepts at key turns:")
    concept_turns = [50, 100, 250, 500]
    for t in concept_turns:
        if t in record_by_turn:
            concepts = record_by_turn[t]["top_5_concepts"]
            print(f"    Turn {t:3d}: {concepts}")

    # 5. prediction error baseline: stage 1 mean vs stage 10 mean
    stage1_errors = [r["prediction_error"] for r in by_stage[1]]
    stage10_errors = [r["prediction_error"] for r in by_stage[10]]
    mean_s1 = np.mean(stage1_errors)
    mean_s10 = np.mean(stage10_errors)

    print(f"\n  Prediction error learning check:")
    print(f"    Stage 1 mean error:  {mean_s1:.4f}")
    print(f"    Stage 10 mean error: {mean_s10:.4f}")
    if mean_s10 < mean_s1:
        print(f"    Result: DECLINING (stage 10 is {(1 - mean_s10/mean_s1)*100:.1f}% "
              f"lower than stage 1)")
    else:
        print(f"    Result: NOT declining (stage 10 >= stage 1)")

    # 6. progress signal check: any non-zero readings by stage 3+?
    stage3plus_progress = []
    for s in range(3, 11):
        for r in by_stage[s]:
            stage3plus_progress.append(r["progress_signal"])

    nonzero_count = sum(1 for p in stage3plus_progress if abs(p) > 1e-6)
    positive_count = sum(1 for p in stage3plus_progress if p > 1e-6)
    print(f"\n  Progress signal check (stages 3-10, {len(stage3plus_progress)} turns):")
    print(f"    Non-zero readings: {nonzero_count}")
    print(f"    Positive readings (model improving): {positive_count}")
    mean_progress = np.mean(stage3plus_progress) if stage3plus_progress else 0.0
    print(f"    Mean progress signal: {mean_progress:+.6f}")

    print(f"\n  Log saved to: {log_path}")
    print('MAX NODES: ', config.MAX_GRAPH_NODES)
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
