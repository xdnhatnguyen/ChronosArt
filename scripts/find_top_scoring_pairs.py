"""scripts/find_top_scoring_pairs.py

Finds the highest-scoring prompt-style pairs from the 200 pairs benchmark
to select for visual comparison in the paper figures.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "output/benchmark_10x20"

def main():
    base_file = OUT_DIR / "records_baseline.csv"
    vmod_file = OUT_DIR / "records_proposed_vmod.csv"
    chrono_file = OUT_DIR / "records_chronosart.csv"

    # Use whatever files currently exist
    target_file = chrono_file if chrono_file.exists() else vmod_file

    if not base_file.exists() or not target_file.exists():
        print("Records files not found yet.")
        return

    df_base = pd.read_csv(base_file)
    df_ours = pd.read_csv(target_file)

    merged = pd.merge(
        df_base, df_ours,
        on=["prompt_id", "style_id"],
        suffixes=("_base", "_ours")
    )

    merged["composite_ours"] = 40.0 * merged["clip_style_ours"] + 35.0 * merged["clip_content_ours"]
    merged["composite_base"] = 40.0 * merged["clip_style_base"] + 35.0 * merged["clip_content_base"]
    merged["delta_composite"] = merged["composite_ours"] - merged["composite_base"]
    merged["delta_style"] = merged["clip_style_ours"] - merged["clip_style_base"]

    print("\n=== TOP 5 PAIRS BY OVERALL COMPOSITE SCORE ===")
    top_overall = merged.sort_values(by="composite_ours", ascending=False).head(5)
    for _, row in top_overall.iterrows():
        print(f"Prompt: {row['prompt_ours']} | Style: {row['style_descriptor_ours']}")
        print(f"  Style Score: {row['clip_style_ours']:.4f} | Content Score: {row['clip_content_ours']:.4f} | Composite: {row['composite_ours']:.2f}")

    print("\n=== TOP 5 PAIRS BY STYLE IMPROVEMENT OVER BASELINE ===")
    top_style_gain = merged.sort_values(by="delta_style", ascending=False).head(5)
    for _, row in top_style_gain.iterrows():
        print(f"Prompt: {row['prompt_ours']} | Style: {row['style_descriptor_ours']}")
        print(f"  Ours Style: {row['clip_style_ours']:.4f} vs Base Style: {row['clip_style_base']:.4f} (Delta: +{row['delta_style']:.4f})")

if __name__ == "__main__":
    main()
