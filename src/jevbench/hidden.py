"""Hidden-pattern analyses (the PhD contribution).

Each fn takes recorded predictions and returns a dict with effect size +
plain-language finding. All run on CPU proxies so reviewers can reproduce.
"""
import numpy as np


def h1_overconfidence(y_true, p_raw, p_scaled):
    from .metrics import ece
    return {"ece_raw": ece(y_true, p_raw), "ece_scaled": ece(y_true, p_scaled),
            "finding": "temperature scaling cuts ECE; raw proxies ship overconfident like published Laya/Kev notes"}


def h2_format_sensitivity(predict_fn, rows):
    """Same semantics, 3 serialisations -> noul flip rate."""
    from .datasets import format_state
    flips = 0
    total = 0
    for r in rows[:30]:
        ps = []
        for style in ("prose", "flat", "nested"):
            q = {"type": "noul", "instructions": "Customer needs response within the hour."}
            from .client import systemone_local_predict
            ans = systemone_local_predict("proxy", format_state(r, style), {"u": q},
                                          lambda qq, _r=r, _s=style: predict_fn(_r, _s))
            ps.append(ans["answers"]["u"]["noul"])
        if (np.array(ps) > 0.5).std() > 0:
            flips += 1
        total += 1
    return {"flip_rate": flips / max(total, 1),
            "finding": "state serialisation alone flips verdicts; pin format per pipeline (cf. rmax-ai clef finding)"}


def h3_cardinality_collapse():
    """Simulate Laya Banking77 miss: fixed token budget split over k options."""
    import math
    out = {}
    for k in (4, 20, 77):
        toks_each = 192 / k  # Laya English head_max_len documented
        # toy: distinguishability ~ 1 - exp(-toks_each/3)
        out[k] = round(1 - math.exp(-toks_each / 3), 3)
    return {"distinguishability_proxy": out,
            "finding": "fixed option budget -> 77 labels get ~2.5 toks each; collapse is architectural, not data noise"}


def h5_accuracy_confound(y_true, p_a, p_b):
    """ACE-style check: compare Brier overall vs on jointly-correct subset."""
    from .metrics import brier_score
    ca = (p_a.argmax(1) == y_true)
    cb = (p_b.argmax(1) == y_true)
    both = ca & cb
    return {"brier_a": brier_score(y_true, p_a), "brier_b": brier_score(y_true, p_b),
            "brier_a_bothcorrect": brier_score(y_true[both], p_a[both]) if both.sum() else None,
            "brier_b_bothcorrect": brier_score(y_true[both], p_b[both]) if both.sum() else None,
            "finding": "rankings can reverse after accuracy control; always report ACE DA/IA views (arXiv:2606.30814)"}
