"""Generate visual comparison between Baseline, Quán quân (Exp 4), and Á quân (Exp 15)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for d in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third-party" / "Infinity", PROJECT_ROOT / "third-party" / "Infinity" / "tools"):
    sys.path.insert(0, str(d))

from chronosart.infinity_runtime import load_runtime_modules, load_bundle
from chronosart.config import ExperimentConfig
from chronosart.style_transfer import StyleTransferEngine
from chronosart.plotting import save_image_tensor

CASES = [
    {
        "id": "kettle_origami",
        "title": "A Kettle (Colorful Origami)",
        "prompt": "a kettle sitting on a wooden table, in colorful origami style",
        "style_image": PROJECT_ROOT / "styles/style_01_colorful_origami.png",
    },
    {
        "id": "cat_glowing",
        "title": "A Cat (Glowing Neon)",
        "prompt": "a cute cat sitting peacefully on a cushion, in glowing neon style",
        "style_image": PROJECT_ROOT / "styles/style_10_glowing_neon.png",
    },
]

CONFIGS = [
    {
        "name": "Baseline (Paper)",
        "tag": "baseline",
        "kwargs": {
            "rank": 8,
            "style_decay": 1.0,
            "style_strength": 1.0,
            "pfb_feature_indices": [2],
            "enable_sac": True,
            "sac_strength": 1.0,
            "sac_head_ratio": 1.0,
            "sac_prediction_start": 2,
            "use_scale_scheduled_tau": False,
            "use_style_token_jitter": False,
        }
    },
    {
        "name": "Quán Quân (Exp 4: Rank 4 + Gentle Decay alpha=0.50)",
        "tag": "champion_exp4",
        "kwargs": {
            "rank": 4,
            "style_decay": 0.50,
            "style_strength": 1.1,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True,
            "sac_strength": 0.75,
            "sac_head_ratio": 0.85,
            "sac_prediction_start": 2,
            "use_scale_scheduled_tau": True,
            "tau_coarse": 0.15,
            "tau_fine_min": 0.85,
            "tau_fine_max": 1.25,
            "top_p_fine": 0.92,
            "use_style_token_jitter": True,
            "jitter_start_step": 5,
            "jitter_dropout": 0.15,
        }
    },
    {
        "name": "Á Quân (Exp 15: Early SAC Start Scale 1)",
        "tag": "runnerup_exp15",
        "kwargs": {
            "rank": 4,
            "style_decay": 0.85,
            "style_strength": 1.1,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True,
            "sac_strength": 0.75,
            "sac_head_ratio": 0.85,
            "sac_prediction_start": 1,
            "use_scale_scheduled_tau": True,
            "tau_coarse": 0.15,
            "tau_fine_min": 0.85,
            "tau_fine_max": 1.25,
            "top_p_fine": 0.92,
            "use_style_token_jitter": True,
            "jitter_start_step": 5,
            "jitter_dropout": 0.15,
        }
    }
]


def main():
    out_dir = PROJECT_ROOT / "output/champions_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    args = parser.parse_args()

    load_runtime_modules()
    bundle = load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn, cfg=args.cfg, tau=args.tau, top_k=args.top_k, top_p=args.top_p, seed=42
    )
    engine = StyleTransferEngine(bundle, config)

    print("Generating visual showcase for Baseline vs Quán Quân vs Á Quân...")
    for case in CASES:
        style_path = case["style_image"]
        style_feats = engine.get_style_features(style_path)

        for cfg in CONFIGS:
            out_path = out_dir / f"{case['id']}_{cfg['tag']}.png"
            print(f"Generating {case['id']} - {cfg['name']}...")
            res = engine.paper_dual_path_generate(
                prompt=case["prompt"],
                style_features=style_feats,
                seed=42,
                **cfg["kwargs"]
            )
            save_image_tensor(res["stylized_image_01"].cpu(), out_path)
            print(f"Saved: {out_path}")

    print("Done! All showcase images saved to output/champions_showcase/")


if __name__ == "__main__":
    main()
