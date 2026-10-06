# Paper outline (for the later PhD paper; this repo is the evidence base)

1. Introduction: System-One (Kahneman fast/slow; TypeSafe Jev Sep 2026) — typed
   choice/score/noul with calibrated probabilities, parallel single pass.
2. Related work: Laya (ModernBERT 421M), Clef/Clef-flash (Qwen 27B/9B + vision),
   Kev (LoRA r16 + pointer head), SemIf/OpenJev logit-read family; JevBench,
   Decision Index; CalArena (Brier-first); ACE (accuracy-controlled calibration);
   JevAdvBench (state-as-untrusted).
3. Method: 4-axis protocol (Accuracy/Brier/ECE/coverage/latency/cost);
   byte-identical states; temperature scaling; bootstrap CIs; shadow protocol
   for hosted APIs (<=12 calls, agreement+TVD+jitter); live states (FX/crypto/
   weather/HN) as real-world distribution shift.
4. Verification verdicts: VERIFIED (Jev/Laya/Clef/Kev/SemIf-family),
   CORRECTED (Coin/B-loss), UNVERIFIED (WaterSheep/CLM-8B/Deem/Imajev/Tev).
5. Results: public tables (this repo) + published third-party numbers clearly
   labelled indicative (never head-to-head without shared items).
6. Hidden patterns H1-H6 (overconfidence, format sensitivity, cardinality
   collapse, score-weakness, ACE reversal, determinism gap).
7. Limitations + ethics: no Jev outputs used for training proxies; live APIs
   billed; adversarial state untrusted; thresholds must be refit per deployment.
8. Reproducibility: docker compose up, fixed seeds, results/*.json, figure registry.
