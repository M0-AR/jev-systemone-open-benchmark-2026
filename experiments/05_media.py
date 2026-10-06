"""05: media assets for README + GitHub Pages (all generated, nothing hand-drawn).

Outputs (verified by this script itself):
  results/dashboard.png     4-panel figure: accuracy / Brier / reliability / coverage
  results/architecture.png  System-One flow diagram (state -> heads -> probabilities)
  docs/demo.mp4             8s narrated-style animated walkthrough (720p, yuv420p)
  docs/demo.gif             640px gif fallback for inline README embedding

Run: PYTHONPATH=src python experiments/05_media.py
Requires: matplotlib, numpy, ffmpeg on PATH.
"""
import json
import pathlib
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"


def load():
    pub = json.loads((RES / "02_public.json").read_text())
    hid = json.loads((RES / "04_hidden.json").read_text())
    return pub, hid


def dashboard(pub, hid):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("System-One open benchmark — reference CPU run (fast profile)", fontsize=13, fontweight="bold")
    models = ["encoder-proxy", "logit-reader"]
    labels = ["Encoder\n(Laya-mech.)", "Logit-reader\n(SemIf-mech.)"]
    color = ["#2563eb", "#f59e0b"]

    t = pub["tickets_choice"]
    ax[0, 0].bar(labels, [t[m]["acc"] for m in models], color=color)
    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].set_title("Accuracy — support tickets (20 test)")
    ax[0, 0].set_ylabel("accuracy")
    n = pub["news4"]
    ax[0, 1].bar(labels, [n[m]["acc"] for m in models], color=color)
    ax[0, 1].set_ylim(0, 1.05)
    ax[0, 1].set_title("Accuracy — news topics (60 test)")

    ax[1, 0].bar(labels, [t[m]["brier"] for m in models], color=color)
    ax[1, 0].set_title("Brier score — tickets (lower is better)")
    ax[1, 0].set_ylabel("Brier")
    h1 = hid["H1_overconfidence"]
    ax[1, 1].bar(["raw", "temperature-\nscaled"], [h1["ece_raw"], h1["ece_scaled"]],
                  color=["#dc2626", "#16a34a"])
    ax[1, 1].set_title("H1: calibration error before/after scaling (lower better)")
    ax[1, 1].set_ylabel("ECE-15")
    for a in ax.flat:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = RES / "dashboard.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("System-One decision model: one state in, typed probabilities out — no text generated",
                 fontsize=12, fontweight="bold")
    boxes = [
        (0.4, 1.0, 2.2, 2.0, "STATE\nticket / FX tick\nweather / HN title", "#dbeafe"),
        (3.4, 1.0, 2.4, 2.0, "ENCODER\n(one forward pass)\nModernBERT / Qwen /\nTF-IDF proxy", "#e0e7ff"),
        (6.6, 2.2, 2.0, 0.8, "choice head\npick 1-of-k", "#fef3c7"),
        (6.6, 1.25, 2.0, 0.8, "score head\nordered levels", "#fef3c7"),
        (6.6, 0.3, 2.0, 0.8, "noul head\nyes/no prob.", "#fef3c7"),
        (9.4, 1.0, 2.2, 2.0, "PROBABILITIES\n+ confidence\ncode branches\non thresholds", "#dcfce7"),
    ]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor="#334155", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    for y in (2.6, 1.65, 0.7):
        ax.add_patch(FancyArrowPatch((5.8, 2.0), (6.6, y), arrowstyle="-|>", mutation_scale=14,
                                     color="#334155"))
    ax.add_patch(FancyArrowPatch((2.6, 2.0), (3.4, 2.0), arrowstyle="-|>", mutation_scale=14, color="#334155"))
    ax.add_patch(FancyArrowPatch((8.6, 2.0), (9.4, 2.0), arrowstyle="-|>", mutation_scale=14, color="#334155"))
    out = RES / "architecture.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def demo_video(pub):
    """Animated walkthrough: accuracy race + Brier + reliability build-up."""
    from matplotlib.animation import FuncAnimation
    t = pub["tickets_choice"]
    n = pub["news4"]
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.suptitle("jev-systemone-open-benchmark-2026 — 60-second tour (8s preview)",
                 fontsize=14, fontweight="bold")
    enc_acc, logit_acc = t["encoder-proxy"]["acc"], t["logit-reader"]["acc"]
    enc_br, logit_br = t["encoder-proxy"]["brier"], t["logit-reader"]["brier"]

    def frame(i):
        for a in ax:
            a.clear()
        p = min(1.0, (i + 1) / 40)
        ax[0].barh(["Encoder-proxy", "Logit-reader"],
                   [enc_acc * p, logit_acc * p], color=["#2563eb", "#f59e0b"])
        ax[0].set_xlim(0, 1.1)
        ax[0].set_title("Step 1 — public benchmark: accuracy (tickets)")
        ax[0].set_xlabel("accuracy")
        q = min(1.0, max(0.0, (i - 35) / 40))
        ax[1].barh(["Encoder-proxy", "Logit-reader"],
                   [enc_br * q, logit_br * q], color=["#2563eb", "#f59e0b"])
        ax[1].set_xlim(0, max(enc_br, logit_br) * 1.15)
        ax[1].set_title("Step 2 — Brier score (lower is better)")
        ax[1].set_xlabel("Brier")
        if i > 80:
            fig.suptitle(f"Step 3 — live data + hidden patterns (news acc "
                         f"{n['encoder-proxy']['acc']:.2f} vs {n['logit-reader']['acc']:.2f})",
                         fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    anim = FuncAnimation(fig, frame, frames=110, interval=70)
    DOCS.mkdir(exist_ok=True)
    mp4 = DOCS / "demo.mp4"
    anim.save(str(mp4), fps=14, dpi=100,
              extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                          "-movflags", "+faststart"])
    plt.close(fig)
    gif = DOCS / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                    "-vf", "scale=640:-1", str(gif)], check=True)
    return mp4, gif


def verify(paths):
    for p in paths:
        assert p.exists() and p.stat().st_size > 0, f"missing/empty: {p}"
    mp4 = DOCS / "demo.mp4"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_name,width,height",
                            "-of", "csv=p=0", str(mp4)],
                           capture_output=True, text=True, check=True)
    print("video streams:", probe.stdout.strip().replace("\n", " | "))
    for p in (RES / "dashboard.png", RES / "architecture.png"):
        subprocess.run(["python3", "-c",
                        "import matplotlib.image as m; m.imread('%s'); print('png-ok %s')"
                        % (p, p.name)], check=True)


def main():
    pub, hid = load()
    made = [dashboard(pub, hid), architecture(), *demo_video(pub)]
    for f in ("dashboard.png", "architecture.png"):
        shutil.copy(RES / f, DOCS / f)
    (DOCS / ".nojekyll").touch()
    verify(made)
    print("media-ok:", [f"{p.name} {p.stat().st_size // 1024}KB" for p in made])


if __name__ == "__main__":
    main()
