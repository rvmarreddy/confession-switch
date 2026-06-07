"""Contrastive acknowledgment steering vector and a swept-magnitude forward hook."""
from __future__ import annotations

import numpy as np


def acknowledgment_direction(activations, acknowledgment, usage):
    """Mean(ack=1) - mean(ack=0) computed WITHIN matched usage strata, so the
    contrast isolates reporting rather than cue usage.
    """
    X = np.asarray(activations, float)
    ack = np.asarray(acknowledgment)
    use = np.asarray(usage)
    diffs = []
    for u in (0, 1):
        m = use == u
        if (ack[m] == 1).any() and (ack[m] == 0).any():
            diffs.append(X[m & (ack == 1)].mean(0) - X[m & (ack == 0)].mean(0))
    if not diffs:
        raise ValueError("need both acknowledged and non-acknowledged rollouts within a usage stratum")
    return np.mean(diffs, axis=0)


def steering_hook(direction, alpha):
    """TransformerLens forward hook: resid += alpha * direction."""
    import torch
    vec = torch.tensor(np.asarray(direction, dtype=float))

    def hook(resid, hook):
        return resid + alpha * vec.to(resid.dtype).to(resid.device)

    return hook
