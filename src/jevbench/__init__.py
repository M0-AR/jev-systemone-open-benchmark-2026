"""jevbench: reproducible System-One decision-model benchmark harness."""
from .metrics import accuracy, brier_score, nll, ece, selective_coverage, bootstrap_ci
from .client import SystemOneRequest, systemone_local_predict
from .models import EncoderProxy, LogitReader, MODEL_CARDS
from .datasets import load_support_tickets, load_20news_subset, format_state
from .live import fetch_live_states

__all__ = [
    "accuracy", "brier_score", "nll", "ece", "selective_coverage", "bootstrap_ci",
    "SystemOneRequest", "systemone_local_predict",
    "EncoderProxy", "LogitReader", "MODEL_CARDS",
    "load_support_tickets", "load_20news_subset", "format_state",
    "fetch_live_states",
]
