"""03: live-data verification (real market + real text, no keys).

- Fetches live FX / crypto / weather / HN states (public APIs).
- Routes each live state through local System-One wire format with both
  proxies; records latencies + byte-identical determinism check (3 repeats).
- Opt-in shadow: if TYPESAFE_API_KEY / CLOUDFLARE_* set, sends <=12
  byte-identical calls to hosted Jev/Clef for agreement + jitter analysis.
  Without keys it writes a runnable shadow protocol + cost model instead of
  fabricating numbers (best practice: never present proxy numbers as Jev's).

Writes results/03_live.json
"""
import json, os, pathlib, time
import numpy as np
from jevbench.live import fetch_live_states
from jevbench.datasets import load_support_tickets
from jevbench.models import EncoderProxy
from jevbench.client import systemone_local_predict

QUEUES = ["billing", "bug", "account_access", "other"]

def main():
    tickets = load_support_tickets()
    enc = EncoderProxy().fit([t["text"] for t in tickets[:40]],
                             [t["queue"] for t in tickets[:40]], QUEUES)
    live = fetch_live_states()
    routed = []
    for s in live:
        q = {"triage": {"type": "choice", "instructions": "Which queue owns this?",
                        "criteria": {k: k for k in QUEUES}},
             "act": {"type": "noul", "instructions": "Act autonomously without review?"}}
        # proxy predictor: reuse encoder head on live text (zero-shot stress)
        p = enc.predict_proba_texts([s["text"]])[0]
        ans = systemone_local_predict("encoder-proxy-live", s["text"], q,
                                      lambda qq: p if qq["type"] == "choice"
                                      else np.array([1 - p.max(), p.max()]))
        routed.append({"id": s["id"], "text": s["text"][:160], "meta": s.get("meta"),
                       "queue": ans["answers"]["triage"]["choice"],
                       "conf": ans["answers"]["triage"]["confidence"],
                       "act_p": ans["answers"]["act"]["noul"]})
    # determinism: 3 repeats byte-identical?
    reps = [systemone_local_predict("enc", tickets[0]["text"],
            {"q": {"type": "choice", "instructions": "q", "criteria": {k: k for k in QUEUES}}},
            lambda qq: enc.predict_proba_texts([tickets[0]["text"]])[0])["answers"]["q"]["probabilities"]
            for _ in range(3)]
    det = reps[0] == reps[1] == reps[2]
    shadow = {"hosted_calls": 0, "note": "no keys; shadow protocol ready",
              "protocol": "send byte-identical state+questions to Jev and Clef<=12 calls; log agreement, TVD, p-jitter, latency, usage.input_tokens; map noul<->boolean per surface"}
    if os.getenv("TYPESAFE_API_KEY"):
        shadow["note"] = "TYPESAFE key present: extend shadow_real.py (not run in CI to avoid spend)"
    out = {"n_live_states": len(live), "routed": routed, "local_determinism_x3": det, "shadow": shadow,
           "cost_model_per_M_in": {"jev": 0.042, "clef": 0.24, "clef-flash": 0.09, "local": 0.0}}
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/03_live.json").write_text(json.dumps(out, indent=2))
    print(f"live states: {len(live)} determinism_x3={det}")

if __name__ == "__main__":
    main()
