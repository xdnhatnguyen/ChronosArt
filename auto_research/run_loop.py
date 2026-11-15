"""auto_research/run_loop.py - Autonomous Loop Evaluator Helper.

Executes the evaluation harness, parses the composite score, logs to results.tsv,
and either keeps the commit or reverts back to the last good state.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSV_PATH = PROJECT_ROOT / "auto_research" / "results.tsv"
LOG_PATH = PROJECT_ROOT / "auto_research" / "run.log"


def get_current_commit() -> str:
    res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def get_best_score_from_tsv() -> float:
    if not TSV_PATH.exists():
        return 0.0
    lines = TSV_PATH.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return 0.0
    best = 0.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 7 and parts[6] == "keep":
            try:
                score = float(parts[1])
                if score > best:
                    best = score
            except ValueError:
                pass
    return best


def log_experiment_tsv(
    commit: str,
    score: float,
    clip_style: float,
    clip_content: float,
    diversity: float,
    mem_gb: float,
    status: str,
    desc: str,
):
    with open(TSV_PATH, "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{score:.4f}\t{clip_style:.4f}\t{clip_content:.4f}\t{diversity:.4f}\t{mem_gb:.1f}\t{status}\t{desc}\n")


def run_evaluation() -> dict:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Launching evaluate.py (fixed budget ~2-3 min)...")
    eval_cmd = [sys.executable, str(PROJECT_ROOT / "auto_research" / "evaluate.py")]
    
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(eval_cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    
    log_text = LOG_PATH.read_text(encoding="utf-8")
    
    # Parse metrics from output
    def extract_val(pattern: str, default: float = 0.0) -> float:
        m = re.search(pattern, log_text, re.MULTILINE)
        return float(m.group(1)) if m else default

    status_match = re.search(r"^status:\s*(\w+)", log_text, re.MULTILINE)
    status = status_match.group(1) if status_match else ("CRASH" if proc.returncode != 0 else "UNKNOWN")

    score = extract_val(r"^composite_score:\s*([\d\.]+)")
    clip_style = extract_val(r"^clip_style:\s*([\d\.]+)")
    clip_content = extract_val(r"^clip_content:\s*([\d\.]+)")
    diversity = extract_val(r"^diversity:\s*([\d\.]+)")
    vram_mb = extract_val(r"^peak_vram_mb:\s*([\d\.]+)")
    mem_gb = vram_mb / 1024.0

    return {
        "status": status,
        "score": score,
        "clip_style": clip_style,
        "clip_content": clip_content,
        "diversity": diversity,
        "mem_gb": mem_gb,
        "returncode": proc.returncode,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", default="manual experiment run", help="Short description of this experiment")
    parser.add_argument("--baseline", action="store_true", help="Mark this run as baseline")
    args = parser.parse_args()

    commit = get_current_commit()
    best_score = get_best_score_from_tsv()
    print(f"Current commit: {commit} | Prior Best Score: {best_score:.4f}")

    res = run_evaluation()
    print(f"Result: status={res['status']} | composite_score={res['score']:.4f} (style={res['clip_style']:.4f}, text={res['clip_content']:.4f}, div={res['diversity']:.4f})")

    if args.baseline:
        action = "keep"
        log_experiment_tsv(commit, res["score"], res["clip_style"], res["clip_content"], res["diversity"], res["mem_gb"], action, "baseline")
        print(f"Recorded BASELINE: {res['score']:.4f}")
        return

    if res["status"] == "SUCCESS" and res["score"] > best_score:
        action = "keep"
        print(f"IMPROVEMENT! Score advanced from {best_score:.4f} -> {res['score']:.4f}. KEEPING COMMIT.")
        log_experiment_tsv(commit, res["score"], res["clip_style"], res["clip_content"], res["diversity"], res["mem_gb"], action, args.desc)
    else:
        action = "discard" if res["status"] == "SUCCESS" else "crash"
        print(f"No improvement (best: {best_score:.4f} vs new: {res['score']:.4f}) or crashed. REVERTING.")
        log_experiment_tsv(commit, res["score"], res["clip_style"], res["clip_content"], res["diversity"], res["mem_gb"], action, args.desc)
        # Revert candidate.py to HEAD
        subprocess.run(["git", "checkout", "HEAD", "auto_research/candidate.py"], cwd=PROJECT_ROOT)
        print("Reverted auto_research/candidate.py to previous state.")


if __name__ == "__main__":
    main()
