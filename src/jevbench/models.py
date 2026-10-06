"""Model proxies under test.

We do NOT claim these ARE Laya/Kev/Clef. They implement the three public
mechanisms those projects document, so anyone can reproduce the qualitative
phenomena on CPU without weights/API keys:

- EncoderProxy (Laya-like, 421M ModernBERT + decision head analogy):
  TF-IDF encoder + LogisticRegression head + temperature scaling.
  Non-autoregressive single forward pass analogy.
- LogitReader (SemIf-like): frozen TF-IDF prototypes, cosine-similarity
  "logit read", softmax. No training of head beyond prototypes.
- PriorBaseline: majority-class / uniform prior. Floor.

Real weights/APIs are addressed in experiments/03_live_verification.py as an
opt-in shadow harness (Jev via TypeSafe, Clef via Workers AI, Kev/Laya via HF),
with byte-identical states and documented noul<->boolean wire gap.

MODEL_CARDS: verified public facts (Oct 2026) for the paper table.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODEL_CARDS = {
    "jev": {"maker": "TypeSafe AI", "weights": "closed API", "params": "undisclosed",
            "context": "64k/req (32k state+longest-q)", "price_per_M_in": 0.042,
            "latency_ms_p50": "236-276 (3rd-party)", "vision": False,
            "api": "POST /v1/systemone {choice,score,noul}"},
    "laya": {"maker": "Convai Innovations", "weights": "Apache-2.0 convaiinnovations/laya",
             "params": "421M (ModernBERT-large 395M + head)", "context": "512/q (1024 typed-decisions)",
             "price_per_M_in": 0.0, "latency_ms_p50": "~33 T4", "vision": False,
             "note": "base ~chance zero-shot; fine-tuned 0.766 vs Jev 0.727 typed-decisions; Banking77 collapse"},
    "clef": {"maker": "Cloudflare Workers AI", "weights": "Apache-2.0 Cloudflare/clef",
            "params": "27B (Qwen3.8-27B post-trained)", "context": "64k",
            "price_per_M_in": 0.24, "latency_ms_p50": "209.3 vendor", "vision": True},
    "clef-flash": {"maker": "Cloudflare", "weights": "Apache-2.0 Cloudflare/clef-flash",
                   "params": "9B (Qwen3.5-9B)", "context": "64k",
                   "price_per_M_in": 0.09, "latency_ms_p50": "38.8 vendor", "vision": True},
    "kev-4b": {"maker": "Jared Palmer", "weights": "Apache-2.0 jaredpalmer/kev-4b",
               "params": "4B base + LoRA r16 + pointer head", "context": "8k served",
               "price_per_M_in": 0.0, "vision": False},
    "kev-9b": {"maker": "Jared Palmer", "weights": "Apache-2.0 jaredpalmer/kev-9b",
               "params": "9B base + LoRA r16 + pointer head", "context": "8k served",
               "price_per_M_in": 0.0, "vision": False},
    "semif": {"maker": "community (TheoLeeCJ/openjev et al.)", "weights": "uses open LLM",
              "params": "frozen LLM + logit read", "price_per_M_in": 0.0, "vision": False},
}


class EncoderProxy:
    """Laya-mechanism proxy: trainable encoder+head, temperature-scaled."""

    def __init__(self, temperature=1.0):
        self.vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=500)
        self.le = LabelEncoder()
        self.temperature = temperature

    def fit(self, texts, labels, options):
        X = self.vec.fit_transform(texts)
        self.options_ = list(options)
        self.opt_index_ = {o: i for i, o in enumerate(self.options_)}
        y = np.array([self.opt_index_[l] for l in labels])
        self.clf.fit(X, y)
        return self

    def fit_temperature(self, texts, labels, temps=(0.5, 1.0, 1.5, 2.0, 3.0)):
        from .metrics import nll
        X = self.vec.transform(texts)
        logits = self.clf.decision_function(X)
        if logits.ndim == 1:
            logits = np.stack([-logits, logits], axis=1)
        # reorder clf columns (sorted classes_) into options order
        logits = logits[:, [list(self.clf.classes_).index(i) for i in range(len(self.options_))]]
        y = np.array([self.opt_index_[l] for l in labels])
        best, best_nll = 1.0, 1e9
        for t in temps:
            p = _softmax(logits / t)
            v = nll(y, p)
            if v < best_nll:
                best, best_nll = t, v
        self.temperature = best
        return best

    def predict_proba_texts(self, texts):
        X = self.vec.transform(texts)
        logits = self.clf.decision_function(X)
        if logits.ndim == 1:
            logits = np.stack([-logits, logits], axis=1)
        logits = logits[:, [list(self.clf.classes_).index(i) for i in range(len(self.options_))]]
        return _softmax(logits / self.temperature)


class LogitReader:
    """SemIf-mechanism proxy: frozen prototypes, cosine 'logit read'."""

    def __init__(self):
        self.vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))

    def fit(self, texts, labels, options):
        X = self.vec.fit_transform(texts).toarray()
        labels = np.asarray(labels)
        self.options_ = list(options)
        self.proto_ = np.stack([
            X[labels == o].mean(axis=0) if (labels == o).any() else np.zeros(X.shape[1])
            for o in self.options_
        ])
        return self

    def predict_proba_texts(self, texts):
        X = self.vec.transform(texts).toarray()
        sim = X @ self.proto_.T  # cosine-ish logit read
        return _softmax(sim * 5.0)


def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _align(p, clf_classes, le_classes):
    # LogisticRegression.classes_ already aligned via LabelEncoder; identity
    return p
