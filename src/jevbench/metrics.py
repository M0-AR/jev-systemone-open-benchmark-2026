"""Metrics following 2026 calibration best practice.

Best practice (verified online, Oct 2026):
- CalArena (arXiv:2605.30188): report Brier (proper score) as primary,
  ECE-15 as secondary; binning-only comparisons mislead.
- ACE (arXiv:2606.30814): raw ECE/Brier confound accuracy; control for
  accuracy before claiming "better calibrated".
- JevBench v1: report 4 axes jointly: Intelligence, Calibration, Speed, Cost.

All functions take y_true (int labels) and y_probs (n x k arrays).
"""
import numpy as np


def accuracy(y_true, y_probs):
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    return float((y_probs.argmax(axis=1) == y_true).mean())


def brier_score(y_true, y_probs):
    """Mean squared error vs one-hot. Lower is better. Primary metric."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n, k = y_probs.shape
    oh = np.zeros_like(y_probs)
    oh[np.arange(n), y_true] = 1.0
    return float(((y_probs - oh) ** 2).sum(axis=1).mean())


def nll(y_true, y_probs, eps=1e-12):
    y_true = np.asarray(y_true)
    y_probs = np.clip(np.asarray(y_probs), eps, 1.0)
    return float(-np.log(y_probs[np.arange(len(y_true)), y_true]).mean())


def ece(y_true, y_probs, n_bins=15):
    """Top-label ECE with equal-width bins. Secondary metric only."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    conf = y_probs.max(axis=1)
    pred = y_probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            out += (m.mean()) * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return float(out)


def selective_coverage(y_true, y_probs, error_budget=0.05):
    """Share of decisions automatable at <= error_budget, gating on confidence.

    Sort by confidence desc, take largest prefix whose error <= budget.
    Returns (coverage, threshold). Mirrors Kev/Jev 'coverage at 5% error'.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    conf = y_probs.max(axis=1)
    correct = (y_probs.argmax(axis=1) == y_true).astype(float)
    order = np.argsort(-conf)
    correct_sorted = correct[order]
    conf_sorted = conf[order]
    best_cov, best_thr = 0.0, 1.0
    for i in range(1, len(order) + 1):
        err = 1.0 - correct_sorted[:i].mean()
        if err <= error_budget + 1e-12:
            best_cov = i / len(order)
            best_thr = float(conf_sorted[i - 1])
        else:
            break
    return float(best_cov), float(best_thr)


def bootstrap_ci(y_true, y_probs, fn=accuracy, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(y_true[idx], y_probs[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
