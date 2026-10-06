"""01: claims verification against live online evidence (no GPU/keys).

Writes results/01_claims.json with VERIFIED / PARTIAL / UNVERIFIED verdicts
and URLs. This is the antidote to hallucinated model cards: every row needs
a primary source fetched or cited from the Oct-2026 verification pass.
"""
import json, pathlib

CLAIMS = [
    {"id": "jev-exists", "claim": "TypeSafe Jev: System-One, choice/score/noul, 70-500ms, $0.042/M, RLCD",
     "verdict": "VERIFIED", "evidence": "https://typesafe.ai/blog/introducing-system-one-models-and-jev + langchain blog + tomshardware 2026-09-21"},
    {"id": "laya-exists", "claim": "Laya 421M ModernBERT-large, Apache-2.0, ~33ms, convaiinnovations/laya",
     "verdict": "VERIFIED", "evidence": "https://huggingface.co/convaiinnovations/laya + zylver benchmark 2026-09-22; caveat: base ~chance zero-shot, fine-tuned 0.766 vs Jev 0.727 typed-decisions; Banking77 collapse documented by vendor"},
    {"id": "clef-exists", "claim": "Cloudflare Clef 27B + Clef-flash 9B, Apache-2.0, vision, 64k, Jev-API compatible, Oct 1 2026",
     "verdict": "VERIFIED", "evidence": "https://blog.cloudflare.com/clef-decision-models/ + developers changelog 2026-10-01 + theregister; vendor table: 7/10 wins but loses When2Call/BRIGHT; $0.24/$0.09 vs $0.042"},
    {"id": "kev-exists", "claim": "Kev family Qwen + LoRA r16 + pointer head, 0.8/4/9/27B, drop-in /v1/systemone",
     "verdict": "VERIFIED", "evidence": "https://github.com/jaredpalmer/kev + HF jaredpalmer/kev-4b,kev-9b; within 1-4pts Jev on new sources; date-arithmetic + MMLU-Pro gaps"},
    {"id": "semif-openjev", "claim": "SemIf logit-read + OpenJev UI/API clones exist as community projects",
     "verdict": "VERIFIED", "evidence": "github TheoLeeCJ/openjev (SemIf-OpenJev), ikermoel/open-alternative-jev, JevBench rows: SemIf 74.6 close to Jev"},
    {"id": "coin-base", "claim": "Clef based on 'Coin 27B / Coin 3.59B' with 'B-loss calibration'",
     "verdict": "UNVERIFIED (corrected)", "evidence": "No primary source. Vendor states Qwen3.8-27B / Qwen3.5-9B + RLCD + B...loss variant. 'Coin' appears to be a hallucination."},
    {"id": "watersheep-clm8b-deem-tev", "claim": "WaterSheep drop-in, CLM-8B Stanford/Nvidia 9x, Deem 9B, Tev 92.8%",
     "verdict": "UNVERIFIED", "evidence": "No primary source found in Oct-2026 pass (web+HN+news+arxiv+HF). Treat as unverified until repo/weights/benchmark published."},
    {"id": "imajev-4b", "claim": "Imajev-4B open decision model (95% figure circulating)",
     "verdict": "PARTIAL", "evidence": "VERIFIED as JevBench entrant: leads v1.4.2.2 (JevBench RELEASE-v1.4.2.2). The viral '95%' figure is UNVERIFIED here (different items/prompts); read the JevBench artifact, not screenshots."},
    {"id": "openjev-groq-cerebras", "claim": "OpenJev natively hooks Groq/Cerebras with 27B near-instant evals",
     "verdict": "PARTIAL", "evidence": "OpenJev-family repos proxy to fast backends, but 'native near-instant 27B evals' is deployment-dependent, not a model property."},
]

def main():
    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    (out / "01_claims.json").write_text(json.dumps(CLAIMS, indent=2))
    print(f"wrote {len(CLAIMS)} claim verdicts -> results/01_claims.json")

if __name__ == "__main__":
    main()
