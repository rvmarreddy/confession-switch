"""Contrastive acknowledgment steering vector and a swept-magnitude forward hook."""
from __future__ import annotations

import numpy as np


def acknowledgment_direction(activations, acknowledgment, usage, answers=None, min_per_cell=10, drop_thin=False):
    """Mean(ack=1) - mean(ack=0) within matched strata, so the contrast isolates reporting
    rather than cue usage or the answer chosen.

    Strata are (usage, chosen answer) as pre-registered; with `answers=None` they fall back
    to usage alone, a weaker match that must be reported as such. A stratum is used only
    if both its cells hold at least `min_per_cell` rollouts. A thin stratum raises, naming
    every thin cell, unless `drop_thin=True`; dropped strata are returned, never silent.
    The stratum differences are averaged with weights equal to stratum size.

    Returns (direction, info) with info = {"used": {stratum: (n_ack, n_not)}, "dropped": {...}}.
    """
    X = np.asarray(activations, float)
    ack = np.asarray(acknowledgment)
    keys = list(zip(usage, answers)) if answers is not None else [(u,) for u in usage]
    used, dropped, diffs, weights = {}, {}, [], []
    for s in sorted(set(keys), key=str):
        m = np.array([k == s for k in keys])
        n1, n0 = int((m & (ack == 1)).sum()), int((m & (ack == 0)).sum())
        if min(n1, n0) < min_per_cell:
            dropped[s] = (n1, n0)
            continue
        used[s] = (n1, n0)
        diffs.append(X[m & (ack == 1)].mean(0) - X[m & (ack == 0)].mean(0))
        weights.append(n1 + n0)
    if dropped and not drop_thin:
        raise ValueError(f"strata below {min_per_cell} per cell (stratum: (n_ack, n_not)): {dropped}; "
                         "collect more rollouts or pass drop_thin=True and report the drop")
    if not diffs:
        raise ValueError(f"no stratum has {min_per_cell} acknowledged and non-acknowledged rollouts")
    return np.average(diffs, axis=0, weights=weights), {"used": used, "dropped": dropped}


def steering_hook(direction, alpha):
    """TransformerLens forward hook: resid += alpha * direction."""
    import torch
    vec = torch.tensor(np.asarray(direction, dtype=float))

    def hook(resid, hook):
        return resid + alpha * vec.to(resid.dtype).to(resid.device)

    return hook
