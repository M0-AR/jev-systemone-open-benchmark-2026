"""02: public-data benchmark (CPU, no keys): EncoderProxy vs LogitReader.

Protocols mirrors JevBench/Decision-Index axes: Accuracy, Brier(primary),
ECE-15(secondary), selective coverage@5%, latency, cost.
Datasets: synthetic support tickets (choice+noul+score) + 20news-4 topic proxy.
"""
import json, time, pathlib
import numpy as np
from jevbench.datasets import load_support_tickets
from jevbench.models import EncoderProxy, LogitReader
from jevbench.metrics import accuracy, brier_score, nll, ece, selective_coverage, bootstrap_ci
from jevbench.client import systemone_local_predict

QUEUES = ["billing", "bug", "account_access", "other"]

def run_tickets(fast=True):
    rows = load_support_tickets()
    texts = [r["text"] for r in rows]
    yq = np.array([QUEUES.index(r["queue"]) for r in rows])
    yu = np.array([r["urgent"] for r in rows])
    # split deterministic
    tr = slice(0, 40); te = slice(40, 60)
    results = {}
    for name, mk in (("encoder-proxy", EncoderProxy), ("logit-reader", LogitReader)):
        m = mk()
        train_labels = [rows[i]["queue"] for i in range(40)]
        m.fit([texts[i] for i in range(40)], train_labels, QUEUES)
        if hasattr(m, "fit_temperature"):
            m.fit_temperature([texts[i] for i in range(40)], train_labels)
        t0 = time.perf_counter()
        p = m.predict_proba_texts([texts[i] for i in range(40, 60)])
        dt = (time.perf_counter() - t0) * 1000 / 20
        lo, hi = bootstrap_ci(yq[40:], p, n_boot=400)
        cov, thr = selective_coverage(yq[40:], p)
        results[name] = {"acc": accuracy(yq[40:], p), "acc_ci95": [lo, hi],
                         "brier": brier_score(yq[40:], p), "nll": nll(yq[40:], p),
                         "ece15": ece(yq[40:], p), "coverage@5%": cov, "thr": thr,
                         "ms_per_q": dt, "cost_per_M": 0.0}
        # wire-format smoke test: Jev SDK swap analogy
        q = {"queue": {"type": "choice", "instructions": "Which queue?", "criteria": {k: k for k in QUEUES}},
             "urgent": {"type": "noul", "instructions": "Needs response within hour."}}
        _ = systemone_local_predict(name, texts[40], q, lambda qq: p[0] if qq["type"] == "choice" else np.array([1 - p[0].max(), p[0].max()]))
    return results

def run_20news(fast=True):
    from jevbench.datasets import load_20news_subset
    trx, try_, tex, tey, names = load_20news_subset(fast=fast)
    names = list(names)
    tr_labels = [names[i] for i in try_]
    te_y = np.asarray(tey)
    out = {}
    for name, mk in (("encoder-proxy", EncoderProxy), ("logit-reader", LogitReader)):
        m = mk().fit(trx, tr_labels, names)
        p = m.predict_proba_texts(tex)
        out[name] = {"acc": accuracy(te_y, p), "brier": brier_score(te_y, p), "ece15": ece(te_y, p)}
    out["labels"] = list(names)
    return out

def main(fast=True):
    r = {"tickets_choice": run_tickets(fast), "news4": run_20news(fast)}
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/02_public.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2)[:2000])

if __name__ == "__main__":
    import sys
    main("--full" not in sys.argv)
