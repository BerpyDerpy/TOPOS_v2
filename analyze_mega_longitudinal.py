#!/usr/bin/env python3
"""Analyze mega-longitudinal run against Experiment 7 hypotheses.

Reads the JSONL log and computes:
  H1: Prediction error transition ratio at music→systems boundary
  H2: Epistemic uncertainty transition ratio (vs Exp 5's 1.05×)
  H3: Progress signal validity within-domain vs at boundary
  H4: Ensemble weight norm divergence at boundary
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def analyze(records):
    print("=" * 70)
    print("EXPERIMENT 7 — HYPOTHESIS EVALUATION")
    print("=" * 70)
    print(f"Total turns: {len(records)}")
    print()

    music = [r for r in records if r["domain"] == "music"]
    systems = [r for r in records if r["domain"] == "systems"]
    print(f"Music turns: {len(music)}")
    print(f"Systems turns: {len(systems)}")
    print()

    # ================================================================
    # H1: Prediction error at domain transition
    # ================================================================
    print("-" * 70)
    print("H1: Prediction error — transition ratio target > 1.10×")
    print("-" * 70)

    # "within-domain baseline" = mean prediction error for last 50 music turns
    # (batch 10, the model is maximally adapted to music)
    music_late = [r["prediction_error"] for r in music[-50:]]
    music_late_mean = np.mean(music_late)

    # "transition" = first 10 systems turns (turns 501-510)
    transition_window = 10
    systems_early = [r["prediction_error"] for r in systems[:transition_window]]
    systems_early_mean = np.mean(systems_early)

    # also compute wider windows
    systems_first_batch = [r["prediction_error"] for r in systems[:50]]
    systems_first_batch_mean = np.mean(systems_first_batch)

    pe_ratio_10 = systems_early_mean / music_late_mean
    pe_ratio_50 = systems_first_batch_mean / music_late_mean

    print(f"  Music last 50 turns (baseline):   mean PE = {music_late_mean:.4f}")
    print(f"  Systems first 10 turns:           mean PE = {systems_early_mean:.4f}")
    print(f"  Systems first 50 turns:           mean PE = {systems_first_batch_mean:.4f}")
    print(f"  Transition ratio (10-turn):  {pe_ratio_10:.4f}×")
    print(f"  Transition ratio (50-turn):  {pe_ratio_50:.4f}×")
    pe_pass = pe_ratio_10 > 1.10
    print(f"  >> H1 {'PASS' if pe_pass else 'FAIL'}: "
          f"{'>' if pe_pass else '<='} 1.10× target")
    print()

    # ================================================================
    # H2: Epistemic uncertainty transition ratio (vs Exp 5's 1.05×)
    # ================================================================
    print("-" * 70)
    print("H2: Epistemic uncertainty — transition ratio improvement over 1.05×")
    print("-" * 70)

    # Within-domain baseline: mean epistemic for music turns 51-500
    # (skip first 50 as cold-start / learning)
    music_stable_ep = [r["epistemic_uncertainty"] for r in music[50:]]
    music_stable_ep_mean = np.mean(music_stable_ep)

    # Late music (last 50, most adapted)
    music_late_ep = [r["epistemic_uncertainty"] for r in music[-50:]]
    music_late_ep_mean = np.mean(music_late_ep)

    # Transition: first 10 and first 50 systems turns
    systems_early_ep = [r["epistemic_uncertainty"] for r in systems[:transition_window]]
    systems_early_ep_mean = np.mean(systems_early_ep)

    systems_first_batch_ep = [r["epistemic_uncertainty"] for r in systems[:50]]
    systems_first_batch_ep_mean = np.mean(systems_first_batch_ep)

    # Within-domain non-transition: systems turns 51-500
    systems_stable_ep = [r["epistemic_uncertainty"] for r in systems[50:]]
    systems_stable_ep_mean = np.mean(systems_stable_ep)

    # Compute ratios against late music baseline
    ep_ratio_10 = systems_early_ep_mean / music_late_ep_mean
    ep_ratio_50 = systems_first_batch_ep_mean / music_late_ep_mean

    # Also compute "transition vs non-transition within systems"
    ep_ratio_internal = systems_first_batch_ep_mean / systems_stable_ep_mean

    print(f"  Music turns 51-500 (stable):      mean epistemic = {music_stable_ep_mean:.6f}")
    print(f"  Music last 50 turns (baseline):    mean epistemic = {music_late_ep_mean:.6f}")
    print(f"  Systems first 10 turns:            mean epistemic = {systems_early_ep_mean:.6f}")
    print(f"  Systems first 50 turns:            mean epistemic = {systems_first_batch_ep_mean:.6f}")
    print(f"  Systems turns 51-500 (stable):     mean epistemic = {systems_stable_ep_mean:.6f}")
    print()
    print(f"  Transition ratio vs late music (10-turn):  {ep_ratio_10:.4f}×")
    print(f"  Transition ratio vs late music (50-turn):  {ep_ratio_50:.4f}×")
    print(f"  Systems batch-1 vs rest (internal):        {ep_ratio_internal:.4f}×")

    ep_pass = ep_ratio_10 > 1.05
    print(f"  >> H2 {'PASS' if ep_pass else 'FAIL'}: "
          f"10-turn transition ratio {ep_ratio_10:.4f}× "
          f"{'>' if ep_pass else '<='} 1.05× (Exp 5 baseline)")
    print()

    # Per-turn epistemic at the boundary (turns 496-510)
    print("  Per-turn epistemic at boundary:")
    boundary_records = [r for r in records if 496 <= r["turn_index"] <= 515]
    for r in boundary_records:
        marker = " <<< TRANSITION" if r["turn_index"] == 501 else ""
        print(f"    turn {r['turn_index']:4d} [{r['domain']:7s}] "
              f"epistemic={r['epistemic_uncertainty']:.6f}  "
              f"error={r['prediction_error']:.4f}{marker}")
    print()

    # ================================================================
    # H3: Progress signal validity
    # ================================================================
    print("-" * 70)
    print("H3: Progress signal — non-artifactual within-domain, "
          "disruption at boundary")
    print("-" * 70)

    # Within-domain stable stretches
    # Music turns 100-500 (skip first 100 for settling)
    music_stable_prog = [r["progress_signal"] for r in music[100:]]
    music_prog_mean = np.mean(music_stable_prog)
    music_prog_std = np.std(music_stable_prog)
    music_prog_positive = sum(1 for p in music_stable_prog if p > 0)
    music_prog_pct_positive = music_prog_positive / len(music_stable_prog) * 100

    # Systems turns 550-1000 (skip first 50 for settling)
    systems_stable_prog = [r["progress_signal"] for r in systems[50:]]
    systems_prog_mean = np.mean(systems_stable_prog)
    systems_prog_std = np.std(systems_stable_prog)
    systems_prog_positive = sum(1 for p in systems_stable_prog if p > 0)
    systems_prog_pct_positive = systems_prog_positive / len(systems_stable_prog) * 100

    # Transition boundary: turns 501-510
    transition_prog = [r["progress_signal"] for r in systems[:transition_window]]
    transition_prog_mean = np.mean(transition_prog)

    # Pre-transition: last 10 music turns
    pre_transition_prog = [r["progress_signal"] for r in music[-10:]]
    pre_transition_prog_mean = np.mean(pre_transition_prog)

    print(f"  Music stable (turns 101-500):")
    print(f"    mean progress = {music_prog_mean:+.4f}")
    print(f"    std progress  = {music_prog_std:.4f}")
    print(f"    positive pct  = {music_prog_pct_positive:.1f}%")
    print(f"  Systems stable (turns 551-1000):")
    print(f"    mean progress = {systems_prog_mean:+.4f}")
    print(f"    std progress  = {systems_prog_std:.4f}")
    print(f"    positive pct  = {systems_prog_pct_positive:.1f}%")
    print(f"  Pre-transition (music last 10):   mean = {pre_transition_prog_mean:+.4f}")
    print(f"  Transition (systems first 10):    mean = {transition_prog_mean:+.4f}")

    # Progress should be disrupted (negative) at boundary
    prog_disrupted = transition_prog_mean < music_prog_mean
    prog_near_zero_music = abs(music_prog_mean) < music_prog_std * 2
    prog_near_zero_systems = abs(systems_prog_mean) < systems_prog_std * 2

    print()
    print(f"  Within-domain near-zero or positive:")
    print(f"    Music:   mean={music_prog_mean:+.4f}, "
          f"|mean| < 2*std ({2*music_prog_std:.4f})? "
          f"{'YES' if prog_near_zero_music else 'NO'}")
    print(f"    Systems: mean={systems_prog_mean:+.4f}, "
          f"|mean| < 2*std ({2*systems_prog_std:.4f})? "
          f"{'YES' if prog_near_zero_systems else 'NO'}")
    print(f"  Boundary disruption:")
    print(f"    Transition mean ({transition_prog_mean:+.4f}) < "
          f"music stable mean ({music_prog_mean:+.4f})? "
          f"{'YES' if prog_disrupted else 'NO'}")

    h3_pass = prog_near_zero_music and prog_disrupted
    print(f"  >> H3 {'PASS' if h3_pass else 'FAIL'}")
    print()

    # ================================================================
    # H4: Ensemble weight norm divergence at boundary
    # ================================================================
    print("-" * 70)
    print("H4: Ensemble weight norm divergence at domain boundary")
    print("-" * 70)

    # Weight norms at specific points
    def get_weight_norms(turn_idx):
        for r in records:
            if r["turn_index"] == turn_idx:
                return r.get("ensemble_weight_norms", None)
        return None

    norms_t1 = get_weight_norms(1)
    norms_t250 = get_weight_norms(250)
    norms_t500 = get_weight_norms(500)
    norms_t501 = get_weight_norms(501)
    norms_t550 = get_weight_norms(550)
    norms_t750 = get_weight_norms(750)
    norms_t1000 = get_weight_norms(1000)

    print("  Weight norms at key points:")
    for label, norms in [("t=1   (init)", norms_t1),
                          ("t=250 (mid-music)", norms_t250),
                          ("t=500 (end-music)", norms_t500),
                          ("t=501 (start-sys)", norms_t501),
                          ("t=550 (batch1-sys)", norms_t550),
                          ("t=750 (mid-sys)", norms_t750),
                          ("t=1000 (end-sys)", norms_t1000)]:
        if norms:
            std = np.std(norms)
            print(f"    {label:25s}  norms={[round(n,2) for n in norms]}  "
                  f"std={std:.4f}")

    # Divergence analysis: std of weight norms over time
    # Sample every 50 turns
    print()
    print("  Weight norm std (divergence) per batch:")
    for i in range(0, len(records), 50):
        r = records[min(i + 49, len(records) - 1)]
        norms = r.get("ensemble_weight_norms")
        if norms:
            std = np.std(norms)
            print(f"    turn {r['turn_index']:4d} [{r['domain']:7s}]  "
                  f"norm_std={std:.4f}  "
                  f"norms=[{', '.join(f'{n:.1f}' for n in norms)}]")

    # Check if divergence is present and growing
    if norms_t1 and norms_t1000:
        init_std = np.std(norms_t1)
        final_std = np.std(norms_t1000)
        divergence_ratio = final_std / max(init_std, 1e-12)
        h4_pass = final_std > init_std
        print()
        print(f"  Init weight norm std:  {init_std:.4f}")
        print(f"  Final weight norm std: {final_std:.4f}")
        print(f"  Divergence ratio:      {divergence_ratio:.2f}×")
        print(f"  >> H4 {'PASS' if h4_pass else 'FAIL'}: "
              f"{'increasing' if h4_pass else 'not increasing'} divergence "
              f"({divergence_ratio:.2f}×)")
    print()

    # ================================================================
    # Per-batch epistemic trajectory (for overview)
    # ================================================================
    print("-" * 70)
    print("EPISTEMIC TRAJECTORY (per-batch means)")
    print("-" * 70)
    for batch_start in range(0, len(records), 50):
        batch = records[batch_start:batch_start + 50]
        ep_mean = np.mean([r["epistemic_uncertainty"] for r in batch])
        pe_mean = np.mean([r["prediction_error"] for r in batch])
        prog_mean = np.mean([r["progress_signal"] for r in batch])
        domain = batch[0]["domain"]
        batch_num = batch_start // 50 + 1
        marker = " <<<" if batch_start == 500 else ""
        print(f"  Batch {batch_num:2d} [{domain:7s}]  "
              f"epistemic={ep_mean:.6f}  "
              f"error={pe_mean:.4f}  "
              f"progress={prog_mean:+.4f}{marker}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    results = {
        "H1 (PE transition > 1.10×)": pe_pass,
        "H2 (Epistemic > 1.05×)": ep_pass,
        "H3 (Progress signal valid)": h3_pass,
        "H4 (Weight norm divergence)": h4_pass,
    }
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(results.values())
    print(f"\n  {passed_count}/4 hypotheses confirmed.")
    print("=" * 70)


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/mega_longitudinal_20260427_093154.jsonl"
    records = load_log(log_path)
    analyze(records)
