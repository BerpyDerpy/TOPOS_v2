# forward_model.py - ensemble forward model and curiosity signal for topos
#
# this is the core curiosity stuff from agents.md section 5:
#
#   EnsembleForwardModel
#     n independent mlps that predict z_{t+1} from concat(z_t, a_t)
#     epistemic uncertainty = mean variance across ensemble members
#
#   CuriositySignal
#     wraps the forward model to get three signals per turn:
#       - epistemic: ensemble disagreement (before seeing ground truth)
#       - error:     prediction error (after seeing ground truth)
#       - progress:  improvement vs rolling mean error
#
#   AdaptiveThreshold
#     dynamic exploration gate: 85th percentile of recent epistemic values
#
# all cpu only. vram is taken by the slm.

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from state_encoder import Z_DIM  # 448


# dimension constants
_ACTION_DIM  = 128               # a_t is in r^128
_INPUT_DIM   = Z_DIM + _ACTION_DIM   # 576 = concat(z_t, a_t)
_OUTPUT_DIM  = Z_DIM                # predict z_{t+1} in r^448
_HIDDEN_DIM  = 256


# =====================
# single forward model member
# =====================

class _ForwardMLP(nn.Module):
    # one mlp of the ensemble
    # architecture: Linear(576->256) -> GELU -> Linear(256->256) -> GELU -> Linear(256->448)
    # no dropout, ensemble diversity comes from different inits and data orderings

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, _HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(_HIDDEN_DIM, _OUTPUT_DIM),
        )
        self._init_weights()

    def _init_weights(self):
        # kaiming for hidden layers, xavier for output
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                if i < len(self.net) - 1:
                    # hidden layers
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
                else:
                    # output layer, smaller init for stable early predictions
                    nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x: (batch, 576) -> (batch, 448)
        return self.net(x)


# =====================
# ensemble forward model
# =====================

class EnsembleForwardModel:
    # n independent mlps predicting z_{t+1} from (z_t, a_t)
    #
    # n_members: how many mlps in the ensemble (default 5)
    # lr: learning rate for each adam optimizer (default 1e-4)

    def __init__(self, n_members=5, lr=1e-4):
        self.n_members = n_members
        self.lr = lr

        self._members = []
        self._optimisers = []

        for _ in range(n_members):
            mlp = _ForwardMLP()
            mlp.eval()  # default to eval, switched to train in update()
            self._members.append(mlp)
            self._optimisers.append(
                torch.optim.Adam(mlp.parameters(), lr=lr)
            )

    # -------
    # main stuff
    # -------

    def predict(self, z_t, a_t):
        # predict z_{t+1} and estimate epistemic uncertainty
        #
        # z_t: shape (448,) current state
        # a_t: shape (128,) action vector
        #
        # returns:
        #   mean_prediction: shape (448,), mean across ensemble
        #   epistemic_uncertainty: float, mean per-dim variance averaged over dims
        x = self._prepare_input(z_t, a_t)

        predictions = []
        with torch.no_grad():
            for mlp in self._members:
                mlp.eval()
                pred = mlp(x).squeeze(0)  # (448,)
                predictions.append(pred)

        # stack to (n, 448)
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)                  # (448,)
        per_dim_var = stacked.var(dim=0, unbiased=False)  # (448,)
        epistemic = per_dim_var.mean().item()

        return mean_pred.numpy(), epistemic

    def update(self, z_t, a_t, z_t1_actual):
        # one online gradient step on all ensemble members
        # each member sees the same sample but takes independent gradient steps
        #
        # z_t: shape (448,)
        # a_t: shape (128,)
        # z_t1_actual: shape (448,), ground truth next state
        #
        # returns mean mse loss across all members
        x = self._prepare_input(z_t, a_t)
        target = torch.from_numpy(
            z_t1_actual.astype(np.float32)
        ).unsqueeze(0)  # (1, 448)

        total_loss = 0.0
        for mlp, opt in zip(self._members, self._optimisers):
            mlp.train()
            pred = mlp(x)                             # (1, 448)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mlp.eval()
            total_loss += loss.item()

        return total_loss / self.n_members

    def update_bagged(self, z_t, a_t, z_t1_actual, keep_prob=0.8):
        # bootstrap aggregating: each member randomly skips this sample
        # with probability (1 - keep_prob). ensures ensemble diversity by
        # giving each member a different effective training set.
        #
        # without bagging, all members converge to nearly identical predictions
        # and epistemic uncertainty collapses to zero everywhere — useless.
        #
        # returns mean mse loss across members that were trained
        x = self._prepare_input(z_t, a_t)
        target = torch.from_numpy(
            z_t1_actual.astype(np.float32)
        ).unsqueeze(0)  # (1, 448)

        total_loss = 0.0
        n_trained = 0

        for mlp, opt in zip(self._members, self._optimisers):
            if np.random.random() > keep_prob:
                continue  # skip this member for this sample
            mlp.train()
            pred = mlp(x)                             # (1, 448)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mlp.eval()
            total_loss += loss.item()
            n_trained += 1

        return total_loss / max(n_trained, 1)

    # -------
    # save/load
    # -------

    def save(self, path):
        # save full ensemble state (weights + optimisers) to disk
        path = Path(path)
        state = {
            "n_members": self.n_members,
            "lr": self.lr,
            "members": [m.state_dict() for m in self._members],
            "optimisers": [o.state_dict() for o in self._optimisers],
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        # load an ensemble from disk
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(n_members=state["n_members"], lr=state["lr"])
        for mlp, sd in zip(model._members, state["members"]):
            mlp.load_state_dict(sd)
        for opt, sd in zip(model._optimisers, state["optimisers"]):
            opt.load_state_dict(sd)
        return model

    # -------
    # internal
    # -------

    def _prepare_input(self, z_t, a_t):
        # concatenate z_t and a_t into a single input tensor
        combined = np.concatenate([z_t, a_t]).astype(np.float32)
        return torch.from_numpy(combined).unsqueeze(0)  # (1, 576)


# =====================
# curiosity signal
# =====================

@dataclass(frozen=True)
class CuriosityState:
    # snapshot of the three curiosity signals for one turn
    #
    # epistemic: ensemble disagreement before seeing z_{t+1}
    # error: l2 norm of (mean_prediction - z_{t+1}_actual)
    # progress: decrease in error vs rolling mean. positive = improving
    epistemic: float
    error: float
    progress: float


class CuriositySignal:
    # wraps EnsembleForwardModel to produce three curiosity signals
    #
    # forward_model: the ensemble
    # window_k: rolling window size for computing progress (default 10)

    def __init__(self, forward_model, window_k=10):
        self._model = forward_model
        self._window_k = window_k
        self._error_history = collections.deque(maxlen=window_k)

    def step(self, z_t, a_t, z_t1_actual):
        # one curiosity measurement cycle:
        # 1. predict z_{t+1} and measure ensemble disagreement
        # 2. compare prediction to ground truth
        # 3. compute learning progress vs recent error history
        # 4. update the forward model with the new sample
        #
        # returns a CuriosityState

        # 1. before seeing ground truth: predict and measure disagreement
        mean_pred, epistemic = self._model.predict(z_t, a_t)

        # 2. prediction error (l2 norm)
        error = float(np.linalg.norm(mean_pred - z_t1_actual))

        # 3. progress: decrease in error vs rolling mean
        if len(self._error_history) > 0:
            rolling_mean = sum(self._error_history) / len(self._error_history)
            progress = rolling_mean - error   # positive = improving
        else:
            progress = 0.0   # no history yet

        # record this error
        self._error_history.append(error)

        # 4. teach the forward model
        self._model.update(z_t, a_t, z_t1_actual)

        return CuriosityState(
            epistemic=epistemic,
            error=error,
            progress=progress,
        )


# =====================
# adaptive threshold
# =====================

class AdaptiveThreshold:
    # dynamic exploration gate based on rolling epistemic uncertainty
    # keeps a window of recent epistemic values and uses the 85th percentile
    # as the threshold. inputs above that are worth exploring.
    #
    # window_size: how many recent values to track (default 50)
    # percentile: what percentile to use (default 85)

    def __init__(self, window_size=50, percentile=85.0):
        self._window_size = window_size
        self._percentile = percentile
        self._values = collections.deque(maxlen=window_size)

    def update(self, epistemic):
        # record a new epistemic value
        self._values.append(epistemic)

    def should_explore(self, epistemic):
        # returns true if the epistemic value is above the threshold
        # before we have enough data (< 2 samples), always explore
        if len(self._values) < 2:
            return True
        threshold = float(np.percentile(list(self._values), self._percentile))
        return epistemic > threshold

    @property
    def threshold(self):
        # current threshold value, or none if not enough data
        if len(self._values) < 2:
            return None
        return float(np.percentile(list(self._values), self._percentile))


# =====================
# standalone test
# =====================
if __name__ == "__main__":
    import sys

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 65)
    print("EnsembleForwardModel standalone simulation test")
    print("=" * 65)

    # synthetic data: two training clusters, one unseen test cluster
    # cluster a: centred around a random point in z-space
    # cluster b: centred around a different random point
    # cluster c: unseen, centred far from both a and b

    centre_a = np.random.randn(Z_DIM).astype(np.float32) * 0.5
    centre_b = np.random.randn(Z_DIM).astype(np.float32) * 0.5 + 2.0
    centre_c = np.random.randn(Z_DIM).astype(np.float32) * 0.5 + 6.0

    def make_sample(centre):
        # generate (z_t, a_t, z_t1) near a cluster centre
        # dynamics are just z_{t+1} = z_t + 0.1 * a_t + small noise
        z_t = centre + np.random.randn(Z_DIM).astype(np.float32) * 0.1
        a_t = np.random.randn(_ACTION_DIM).astype(np.float32) * 0.3
        z_t1 = z_t + 0.1 * np.concatenate([
            a_t, np.zeros(Z_DIM - _ACTION_DIM, dtype=np.float32)
        ]) + np.random.randn(Z_DIM).astype(np.float32) * 0.02
        return z_t, a_t, z_t1

    # train on 100 samples alternating between clusters a and b
    print("\nPhase 1: Training on clusters A and B (100 steps)...")

    model = EnsembleForwardModel(n_members=5, lr=1e-3)  # higher lr for fast demo
    curiosity = CuriositySignal(model, window_k=10)
    threshold = AdaptiveThreshold(window_size=50, percentile=85.0)

    training_losses = []
    for step in range(100):
        centre = centre_a if step % 2 == 0 else centre_b
        z_t, a_t, z_t1 = make_sample(centre)
        state = curiosity.step(z_t, a_t, z_t1)
        threshold.update(state.epistemic)
        training_losses.append(state.error)
        if (step + 1) % 25 == 0:
            recent = training_losses[-10:]
            print(f"  step {step+1:3d}  |  "
                  f"error={state.error:.4f}  "
                  f"epistemic={state.epistemic:.6f}  "
                  f"progress={state.progress:+.4f}  "
                  f"mean_recent_error={sum(recent)/len(recent):.4f}")

    # measure uncertainty on training dist vs unseen cluster
    print("\nPhase 2: Comparing uncertainty seen vs. unseen clusters...")

    n_test = 30

    # seen clusters (a and b)
    seen_uncertainties = []
    for _ in range(n_test):
        centre = centre_a if np.random.random() > 0.5 else centre_b
        z_t, a_t, _ = make_sample(centre)
        _, unc = model.predict(z_t, a_t)
        seen_uncertainties.append(unc)

    # unseen cluster (c)
    unseen_uncertainties = []
    for _ in range(n_test):
        z_t, a_t, _ = make_sample(centre_c)
        _, unc = model.predict(z_t, a_t)
        unseen_uncertainties.append(unc)

    mean_seen   = np.mean(seen_uncertainties)
    mean_unseen = np.mean(unseen_uncertainties)
    ratio = mean_unseen / max(mean_seen, 1e-12)

    print(f"\n  Mean epistemic (seen)   : {mean_seen:.6f}")
    print(f"  Mean epistemic (unseen) : {mean_unseen:.6f}")
    print(f"  Ratio (unseen/seen)     : {ratio:.2f}x")

    # validation assertions
    print("\nPhase 3: Validation assertions...")

    # unseen should be more uncertain than seen
    assert mean_unseen > mean_seen, (
        f"FAILED: Expected unseen uncertainty ({mean_unseen:.6f}) > "
        f"seen uncertainty ({mean_seen:.6f}).  The ensemble is not "
        f"distinguishing in- from out-of-distribution inputs."
    )
    print(f"  ok  Unseen uncertainty > seen uncertainty "
          f"({mean_unseen:.6f} > {mean_seen:.6f})")

    # progress should eventually become positive (model improving)
    # check last 20 steps of training
    late_progress = []
    for _ in range(20):
        centre = centre_a if np.random.random() > 0.5 else centre_b
        z_t, a_t, z_t1 = make_sample(centre)
        state = curiosity.step(z_t, a_t, z_t1)
        late_progress.append(state.progress)
    # at least some progress values should be positive
    positive_count = sum(1 for p in late_progress if p > 0)
    print(f"  ok  Positive progress in {positive_count}/20 late steps")

    # adaptive threshold should flag unseen inputs
    explore_seen = 0
    for _ in range(n_test):
        centre = centre_a if np.random.random() > 0.5 else centre_b
        z_t, a_t, _ = make_sample(centre)
        _, unc = model.predict(z_t, a_t)
        if threshold.should_explore(unc):
            explore_seen += 1

    explore_unseen = 0
    for _ in range(n_test):
        z_t, a_t, _ = make_sample(centre_c)
        _, unc = model.predict(z_t, a_t)
        if threshold.should_explore(unc):
            explore_unseen += 1

    print(f"  Threshold triggers (seen)   : {explore_seen}/{n_test}")
    print(f"  Threshold triggers (unseen) : {explore_unseen}/{n_test}")
    assert explore_unseen > explore_seen, (
        f"FAILED: Expected more threshold triggers for unseen "
        f"({explore_unseen}) than seen ({explore_seen})."
    )
    print(f"  ok  More exploration triggers for unseen inputs")

    # save/load roundtrip test
    print("\nPhase 4: save/load roundtrip...")

    save_path = Path(__file__).parent / "_test_ensemble.pt"
    model.save(save_path)

    loaded = EnsembleForwardModel.load(save_path)
    z_t, a_t, _ = make_sample(centre_a)
    pred_orig, unc_orig = model.predict(z_t, a_t)
    pred_load, unc_load = loaded.predict(z_t, a_t)

    assert np.allclose(pred_orig, pred_load, atol=1e-6), \
        "Predictions differ after load"
    assert abs(unc_orig - unc_load) < 1e-8, \
        "Uncertainty differs after load"
    print(f"  ok  Save/load roundtrip: predictions match")

    # clean up test file
    save_path.unlink()
    print(f"  ok  Test checkpoint cleaned up")

    # shape and type checks
    print("\nPhase 5: shape and type validation...")

    z_t, a_t, z_t1 = make_sample(centre_a)
    pred, unc = model.predict(z_t, a_t)
    assert pred.shape == (Z_DIM,), f"predict shape: {pred.shape}"
    assert isinstance(unc, float), f"uncertainty type: {type(unc)}"
    print(f"  ok  predict() -> (ndarray {pred.shape}, float)")

    state = curiosity.step(z_t, a_t, z_t1)
    assert isinstance(state, CuriosityState)
    assert isinstance(state.epistemic, float)
    assert isinstance(state.error, float)
    assert isinstance(state.progress, float)
    print(f"  ok  step() -> CuriosityState(epistemic={state.epistemic:.6f}, "
          f"error={state.error:.4f}, progress={state.progress:+.4f})")

    print("\n" + "-" * 65)
    print("All assertions passed.")
    print("=" * 65)
