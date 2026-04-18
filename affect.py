# affect.py - the feelings module for topos
# keeps track of a vector that accumulates surprise and sentiment
# over time, and you can read arousal and valence off it

import numpy as np

from config import (
    AFFECT_DIM, AFFECT_ALPHA, AFFECT_DECAY,
    AROUSAL_HIGH, AROUSAL_MODERATE,
    VALENCE_POS, VALENCE_NEG,
)


class AffectiveState:
    # tracks internal affect as a numpy vector
    # lives in R^AFFECT_DIM, gets blended with new stuff each turn
    # and decays back toward zero over time

    def __init__(self):
        self._state = np.zeros(AFFECT_DIM, dtype=float)

    # -------
    # the main stuff
    # -------

    def update(self, surprise, sentiment):
        # take in a new surprise/sentiment pair and blend it in
        # surprise should be 0 to 1, sentiment should be -1 to 1

        # make a stimulus vector - fill everything with surprise
        # then multiply the first half by sentiment so we can
        # read valence from the leading slice later
        stimulus = np.full(AFFECT_DIM, surprise, dtype=float)
        half = AFFECT_DIM // 2
        stimulus[:half] *= sentiment  # first half is the signed valence part

        # decay toward zero then blend in new stuff
        self._state = AFFECT_DECAY * self._state + AFFECT_ALPHA * stimulus

    def arousal(self):
        # l2 norm of state, normalised to 0-1
        # max possible norm is sqrt(AFFECT_DIM) for a unit-filled vector
        raw = float(np.linalg.norm(self._state))
        max_norm = float(np.sqrt(AFFECT_DIM))
        return min(raw / max_norm, 1.0) if max_norm > 0 else 0.0

    def valence(self):
        # mean of the first half of the state vector, clipped to -1,1
        half = AFFECT_DIM // 2
        raw = float(np.mean(self._state[:half]))
        return float(np.clip(raw, -1.0, 1.0))

    def summary(self):
        # returns something like "arousal: high, valence: positive"
        a = self.arousal()
        v = self.valence()

        if a > AROUSAL_HIGH:
            arousal_label = "high"
        elif a > AROUSAL_MODERATE:
            arousal_label = "moderate"
        else:
            arousal_label = "low"

        if v > VALENCE_POS:
            valence_label = "positive"
        elif v < VALENCE_NEG:
            valence_label = "negative"
        else:
            valence_label = "neutral"

        return f"arousal: {arousal_label}, valence: {valence_label}"


# quick test
if __name__ == "__main__":
    affect = AffectiveState()

    print("Initial state:")
    print(f"  arousal={affect.arousal():.4f}, valence={affect.valence():.4f}")
    print(f"  summary = {affect.summary()!r}")
    print()

    # throw some events at it and see what happens
    events = [
        (0.9, 0.8),   # high surprise, positive
        (0.7, 0.6),
        (0.5, 0.4),
        (0.2, -0.5),  # low surprise, negative
        (0.1, -0.9),  # barely any surprise, very negative
    ]

    for i, (surprise, sentiment) in enumerate(events, 1):
        affect.update(surprise, sentiment)
        print(
            f"After event {i} (surprise={surprise}, sentiment={sentiment:+.1f}): "
            f"arousal={affect.arousal():.4f}, valence={affect.valence():.4f} "
            f"= {affect.summary()!r}"
        )
