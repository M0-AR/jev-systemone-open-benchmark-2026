"""Public datasets: reproducible, no keys, small and licence-clean.

- support_tickets: 60 hand-written synthetic tickets (billing/bug/access/other
  + urgent noul + severity score). Mirrors TypeSafe docs example
  ("charged twice... refund") and rmax-ai/jev-vs-clef battery shape.
- 20news subset: sklearn fetch_20newsgroups (4 classes, capped) as AG-News-like
  topic proxy. Cached under data/; --fast uses 120 docs, --full 800.

format_state(style): tests H2 format sensitivity (prose vs flat JSON vs
nested JSON) found in live Jev/Clef harness (rmax-ai): same semantics,
different serialisation.
"""
import json
from functools import lru_cache

TICKETS = [
    ("I was charged twice for my September invoice, refund one payment.", "billing", 1, 2),
    ("Login fails with SSO error 500 since this morning.", "account_access", 1, 2),
    ("App crashes when exporting PDF with images.", "bug", 0, 1),
    ("How do I reset my password?", "account_access", 0, 0),
    ("Invoice #4412 has wrong VAT number, please reissue.", "billing", 0, 1),
    ("Server returns 502 on checkout, losing orders!", "bug", 1, 2),
    ("Refund my annual plan, I cancelled last week.", "billing", 1, 1),
    ("Cannot invite teammates, permission denied.", "account_access", 0, 1),
    ("Feature request: dark mode for dashboard.", "other", 0, 0),
    ("You overcharged me again, fix this now!", "billing", 1, 2),
] * 6  # 60 rows, deterministic repetition with index-varying phrasing


def load_support_tickets():
    rows = []
    for i, (text, queue, urgent, sev) in enumerate(TICKETS):
        rows.append({
            "id": f"t{i:03d}",
            "text": f"{text} (case {i})",
            "queue": queue,
            "urgent": urgent,
            "severity": sev,  # 0 low, 1 med, 2 high
        })
    return rows


def format_state(row, style="prose"):
    if style == "prose":
        return row["text"]
    if style == "flat":
        return json.dumps({"text": row["text"], "id": row["id"]})
    if style == "nested":
        return json.dumps({"ticket": {"body": {"text": row["text"]}, "meta": {"id": row["id"]}}})
    raise ValueError(style)


def load_20news_subset(n_train=400, n_test=200, fast=True):
    from sklearn.datasets import fetch_20newsgroups
    cats = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.mideast"]
    if fast:
        n_train, n_test = 120, 60
    tr = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
    te = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))
    return (tr.data[:n_train], tr.target[:n_train], te.data[:n_test], te.target[:n_test], tr.target_names)
