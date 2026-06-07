"""Linear probes for usage vs acknowledgment, with AUROC and direction cosine."""
from __future__ import annotations

import numpy as np


def mean_diff_direction(X, y):
    """Difference-of-means probe direction. X: (n, d), y: (n,) in {0,1}."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def auroc(scores, y):
    s = np.asarray(scores, float)
    y = np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return float(wins / (pos.size * neg.size))


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def two_probe_report(activations, usage, acknowledgment):
    """AUROCs of each probe and the cosine between the two probe directions."""
    X = np.asarray(activations, float)
    d_use = mean_diff_direction(X, usage)
    d_ack = mean_diff_direction(X, acknowledgment)
    return {
        "auroc_usage": auroc(X @ d_use, usage),
        "auroc_ack": auroc(X @ d_ack, acknowledgment),
        "cosine_use_ack": cosine(d_use, d_ack),
        "d_use": d_use,
        "d_ack": d_ack,
    }
