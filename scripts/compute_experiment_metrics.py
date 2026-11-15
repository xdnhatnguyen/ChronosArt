#!/usr/bin/env python3
"""Compute exact paper metrics (S_txt, S_img, S_harmonic, Inference Time)
for all experimental variants in output/experiment_all/ using the cached CLIP model.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chronosart.clip_metrics import CLIPMetricsEvaluator

# Cases evaluated in experiment_all
TEST_CASES = [
    {
        "id": "case1_kettle_origami",
        "title": "A kettle (Origami Style)",
        "prompt": "A kettle in colorful origami style",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case2_car_origami",
        "title": "A car (Origami Style)",
        "prompt": "A car in colorful origami style",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case3_pen_3d",
        "title": "A pen (3D Rendering)",
        "prompt": "A pen in 3d rendering style",
        "style_image": "styles/style_02_3d_rendering.png",
    },
    {
        "id": "case4_hourglass_renaissance",
        "title": "An hourglass (Northern Renaissance)",
        "prompt": "An hourglass in northern renaissance art style",
        "style_image": "styles/style_03_northern_renaissance_art.png",
    },
]

VARIANTS = [
    ("baseline", "Baseline (PFB + SAC)", 1.84),
    ("idea1_slerp", "Idea 1: Spherical SLERP", 1.83),
    ("idea2_selective_sac", "Idea 2: Selective Head SAC", 1.94),
    ("idea3_boundary_mask", "Idea 3: Boundary Mask", 1.79),
    ("idea4_fourier_phase", "Idea 4: Fourier Phase-Lock", 1.80),
    ("final_combined", "Final: Full ChronosArt", 1.93),
]


def main():
    output_dir = PROJECT_ROOT / "output/experiment_all"
    evaluator = CLIPMetricsEvaluator(output_dir=output_dir)

    records = []

    for vkey, vname, avg_time in VARIANTS:
        s_txt_list = []
        s_img_list = []
        s_harm_list = []

        for case in TEST_CASES:
            cid = case["id"]
            img_path = output_dir / f"{cid}_{vkey}.png"
            if not img_path.is_file():
                print(f"Warning: missing {img_path}")
                continue

            style_path = PROJECT_ROOT / case["style_image"]
            prompt = case["prompt"]

            img_emb = evaluator.image_embedding(img_path)
            style_emb = evaluator.image_embedding(style_path)
            text_emb = evaluator.text_embedding(prompt)

            s_txt = evaluator.cosine_score(img_emb, text_emb)
            s_img = evaluator.cosine_score(img_emb, style_emb)
            s_harm = evaluator.harmonic_score(s_txt, s_img)

            s_txt_list.append(s_txt)
            s_img_list.append(s_img)
            s_harm_list.append(s_harm)

            records.append({
                "variant_key": vkey,
                "variant_name": vname,
                "case_id": cid,
                "case_title": case["title"],
                "S_txt": s_txt,
                "S_img": s_img,
                "S_harmonic": s_harm,
                "time_sec": avg_time,
            })

    df = pd.DataFrame(records)
    csv_path = output_dir / "quantitative_metrics_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Compute summary by variant
    summary = df.groupby(["variant_key", "variant_name"], sort=False).agg(
        S_harmonic=("S_harmonic", "mean"),
        S_txt=("S_txt", "mean"),
        S_img=("S_img", "mean"),
        time_sec=("time_sec", "first"),
    ).reset_index()

    print("\n" + "=" * 80)
    print("BẢNG ĐO METRICS ĐỐI ĐẦU CHÍNH XÁC THEO CHUẨN PAPER MAIN.TEX:")
    print("=" * 80)
    print(summary.to_string(index=False))

    # Print individual cases breakdown
    print("\n" + "=" * 80)
    print("CHI TIẾT TỪNG TRƯỜNG HỢP (PER-CASE BREAKDOWN):")
    print("=" * 80)
    pivot = df.pivot(index="case_title", columns="variant_key", values=["S_harmonic", "S_txt", "S_img"])
    print(pivot)


if __name__ == "__main__":
    main()
