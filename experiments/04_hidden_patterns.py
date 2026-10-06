"""04: hidden-pattern mining (reproducible, CPU).

H1 overconfidence, H2 format sensitivity, H3 cardinality collapse,
H5 accuracy-confound (ACE-style). Writes results/04_hidden.json +
results/hidden.png (reliability diagram proxy).
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jevbench.datasets import load_support_tickets, format_state
from jevbench.models import EncoderProxy, LogitReader
from jevbench.hidden import h1_overconfidence, h3_cardinality_collapse, h5_accuracy_confound
from jevbench.client import systemone_local_predict

QUEUES = ["billing", "bug", "account_access", "other"]

def main():
    rows = load_support_tickets()
    texts = [r["text"] for r in rows]
    y = np.array([QUEUES.index(r["queue"]) for r in rows])
    train_labels = [r["queue"] for r in rows[:40]]
    enc = EncoderProxy().fit(texts[:40], train_labels, QUEUES)
    raw = enc.predict_proba_texts(texts[40:])
    enc.fit_temperature(texts[:40], train_labels)
    scaled = enc.predict_proba_texts(texts[40:])
    h1 = h1_overconfidence(y[40:], raw, scaled)
    # H2: format flips via a format-coupled noul proxy (serialisation artifact latch).
    # Honest toy: predictor keys off JSON braces, mimicking a model that overfits
    # formatting instead of semantics (cf. rmax-ai nested-JSON 0.09 vs prose 0.90).
    def brittle(row, style):
        s = format_state(row, style)
        p = 0.9 if style == "prose" else (0.15 if style == "flat" else 0.85)
        # nested JSON is long; flat JSON has braces but shallow -> low p
        return np.array([1 - p, p])
    flips = 0
    for r in rows[:30]:
        from jevbench.hidden import h2_format_sensitivity
        break
    from jevbench.hidden import h2_format_sensitivity
    h2 = h2_format_sensitivity(brittle, rows)
    h3 = h3_cardinality_collapse()
    lr = LogitReader().fit(texts[:40], train_labels, QUEUES)
    plr = lr.predict_proba_texts(texts[40:])
    h5 = h5_accuracy_confound(y[40:], scaled, plr)
    out = {"H1_overconfidence": h1, "H2_format_sensitivity_toy": h2,
           "H3_cardinality": h3, "H5_accuracy_confound": h5}
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/04_hidden.json").write_text(json.dumps(out, indent=2))
    # reliability diagram for scaled proxy
    conf = scaled.max(1); pred = scaled.argmax(1); yt = y[40:]
    bins = np.linspace(0, 1, 11); xs, ys = [], []
    for i in range(10):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            xs.append(conf[m].mean()); ys.append((pred[m] == yt[m]).mean())
    plt.figure(); plt.plot([0, 1], [0, 1], "--"); plt.scatter(xs, ys)
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("Reliability (encoder-proxy, scaled)")
    plt.savefig("results/hidden.png", dpi=120)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
