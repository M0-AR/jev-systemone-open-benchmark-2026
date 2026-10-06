"""run_all: zero-to-hero pipeline. Verifies each step by execution."""
import argparse, subprocess, sys
STEPS = ["experiments/01_claims_verification.py", "experiments/02_public_benchmark.py",
         "experiments/03_live_verification.py", "experiments/04_hidden_patterns.py",
         "experiments/05_media.py"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    for s in STEPS:
        cmd = [sys.executable, s] + ([] if a.full else ["--fast"])
        print(f"\n=== {s} ===")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\nALL STEPS OK. See results/*.json + results/hidden.png")

if __name__ == "__main__":
    main()
