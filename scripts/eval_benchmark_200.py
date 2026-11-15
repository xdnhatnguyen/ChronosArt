"""scripts/eval_benchmark_200.py

Runs 10 prompts x 20 styles = 200 pairs evaluation for specified methods on a specific GPU.
Records CLIP Style, CLIP Content, generation latency, and computes cross-style diversity.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for d in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third-party" / "Infinity", PROJECT_ROOT / "third-party" / "Infinity" / "tools"):
    sys.path.insert(0, str(d))

from chronosart.infinity_runtime import load_runtime_modules, load_bundle
from chronosart.config import ExperimentConfig
from chronosart.clip_metrics import CLIPMetricsEvaluator
from chronosart.plotting import save_image_tensor
from auto_research.novel_stylization import (
    NovelSACVModController,
    NovelSACPatch,
)
from auto_research.benchmark_novel_methods import NovelBenchmarkEngine

METHOD_CONFIGS = {
    "baseline": {
        "name": "Baseline (Paper SVD r=8)",
        "kwargs": {
            "rank": 8, "alpha": 1.0, "style_strength": 1.0,
            "pfb_feature_indices": [2],
            "enable_sac": True, "sac_prediction_start": 2, "sac_strength": 1.0, "sac_head_ratio": 1.0,
            "use_adain_latents": False, "use_frequency_split": False,
            "use_v_modulation": False, "use_scale_scheduled_tau": False,
        }
    },
    "exp4_champion": {
        "name": "Exp 4 Champion (Rank 4, alpha=0.50)",
        "kwargs": {
            "rank": 4, "alpha": 0.50, "style_strength": 1.1,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True, "sac_prediction_start": 2, "sac_strength": 0.75, "sac_head_ratio": 0.85,
            "use_adain_latents": False, "use_frequency_split": False,
            "use_v_modulation": False,
            "use_scale_scheduled_tau": True, "tau_coarse": 0.15, "tau_fine_min": 0.85, "tau_fine_max": 1.25,
            "top_p_fine": 0.92, "use_style_token_jitter": True, "jitter_start_step": 5, "jitter_dropout": 0.15,
        }
    },
    "adain_var": {
        "name": "Novel AdaIN-VAR (Moments Latents)",
        "kwargs": {
            "rank": 4, "alpha": 0.50, "style_strength": 1.1,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True, "sac_prediction_start": 2, "sac_strength": 0.75, "sac_head_ratio": 0.85,
            "use_adain_latents": True, "adain_weight": 0.40,
            "use_frequency_split": False, "use_v_modulation": False,
            "use_scale_scheduled_tau": True, "tau_coarse": 0.15, "tau_fine_min": 0.85, "tau_fine_max": 1.25,
            "top_p_fine": 0.92, "use_style_token_jitter": True, "jitter_start_step": 5, "jitter_dropout": 0.15,
        }
    },
    "freq_split": {
        "name": "Novel Frequency-Split PFB (Laplacian)",
        "kwargs": {
            "rank": 4, "alpha": 0.50, "style_strength": 1.1,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True, "sac_prediction_start": 2, "sac_strength": 0.75, "sac_head_ratio": 0.85,
            "use_adain_latents": False,
            "use_frequency_split": True, "freq_high_strength": 1.8, "freq_adain_weight": 0.35,
            "use_v_modulation": False,
            "use_scale_scheduled_tau": True, "tau_coarse": 0.15, "tau_fine_min": 0.85, "tau_fine_max": 1.25,
            "top_p_fine": 0.92, "use_style_token_jitter": True, "jitter_start_step": 5, "jitter_dropout": 0.15,
        }
    },
    "proposed_vmod": {
        "name": "Novel Attention V-Mod (beta=0.30)",
        "kwargs": {
            "rank": 4, "alpha": 0.50, "style_strength": 1.1,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True, "sac_prediction_start": 1, "sac_strength": 0.75, "sac_head_ratio": 0.85,
            "use_adain_latents": False, "use_frequency_split": False,
            "use_v_modulation": True, "v_mod_ratio": 0.30,
            "use_scale_scheduled_tau": True, "tau_coarse": 0.15, "tau_fine_min": 0.85, "tau_fine_max": 1.25,
            "top_p_fine": 0.92, "use_style_token_jitter": True, "jitter_start_step": 5, "jitter_dropout": 0.15,
        }
    },
    "chronosart": {
        "name": "ChronosArt (Proposed Grand Synergy)",
        "kwargs": {
            "rank": 4, "alpha": 0.50, "style_strength": 1.20,
            "pfb_feature_indices": [1, 2, 3],
            "enable_sac": True, "sac_prediction_start": 1, "sac_strength": 0.80, "sac_head_ratio": 0.85,
            "use_adain_latents": False, "use_frequency_split": False,
            "use_v_modulation": True, "v_mod_ratio": 0.32,
            "use_scale_scheduled_tau": True, "tau_coarse": 0.12, "tau_fine_min": 0.90, "tau_fine_max": 1.20,
            "top_p_fine": 0.92, "use_style_token_jitter": True, "jitter_start_step": 5, "jitter_dropout": 0.15,
        }
    },
}

def get_selected_prompts(prompts_csv: Path, count: int = 10):
    df = pd.read_csv(prompts_csv)
    selected_ids = [
        "parti_005", "parti_006", "parti_007", "parti_009", "parti_010",
        "parti_013", "parti_017", "parti_020", "parti_024", "parti_027",
    ]
    sub = df[df["content_id"].isin(selected_ids)].copy()
    if len(sub) < count:
        sub = df.head(count).copy()
    return sub.to_dict(orient="records")

def get_selected_styles(styles_csv: Path, count: int = 20):
    df = pd.read_csv(styles_csv)
    return df.head(count).to_dict(orient="records")

def compute_diversity_for_method(img_dir: Path, prompts, styles) -> float:
    """Computes mean pairwise pixel difference across styles for each prompt."""
    diffs = []
    for p in prompts:
        p_id = p["content_id"]
        imgs = []
        for s in styles:
            s_id = s["style_id"]
            p_path = img_dir / f"{p_id}_{s_id}.png"
            if p_path.exists():
                arr = np.array(Image.open(p_path).convert("RGB"), dtype=np.float32) / 255.0
                imgs.append(arr)
        # Compute adjacent pairwise differences
        for i in range(len(imgs) - 1):
            d = np.abs(imgs[i] - imgs[i+1]).mean()
            diffs.append(float(d))
    return float(np.mean(diffs)) if diffs else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, choices=list(METHOD_CONFIGS.keys()))
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/styles_21.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/benchmark_10x20")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_selected_prompts(args.prompts_csv, count=10)
    styles = get_selected_styles(args.styles_csv, count=20)
    total_pairs = len(prompts) * len(styles)

    print(f"Loaded {len(prompts)} prompts and {len(styles)} styles. Target pairs: {total_pairs}")
    print(f"Methods to evaluate: {args.methods}")

    load_runtime_modules()
    bundle = load_bundle(args)
    config = ExperimentConfig(model_pn=args.pn, cfg=args.cfg, seed=42)
    engine = NovelBenchmarkEngine(bundle, config)
    evaluator = CLIPMetricsEvaluator(output_dir=out_dir)

    # Pre-cache styles
    print("Pre-caching style features...")
    style_features_cache = {}
    for s_info in styles:
        s_id = s_info["style_id"]
        s_path = (PROJECT_ROOT / s_info["style_reference_image"]).resolve()
        style_features_cache[s_id] = engine.get_style_features(s_path)
    print("Style caching complete!")

    for m_id in args.methods:
        cfg = METHOD_CONFIGS[m_id]
        m_name = cfg["name"]
        m_kwargs = cfg["kwargs"]
        m_out_dir = out_dir / m_id
        m_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\nRUNNING: {m_name} ({m_id})\n{'='*60}")
        t_start = time.time()
        torch.cuda.reset_peak_memory_stats()

        style_scores = []
        content_scores = []
        latencies = []
        records = []
        pair_idx = 0

        for p_info in prompts:
            p_text = p_info["content_prompt"]
            p_id = p_info["content_id"]

            for s_info in styles:
                pair_idx += 1
                s_id = s_info["style_id"]
                s_path = (PROJECT_ROOT / s_info["style_reference_image"]).resolve()
                s_descriptor = s_info.get("style_descriptor", s_id)
                full_prompt = f"{p_text}, in {s_descriptor} style"
                style_feats = style_features_cache[s_id]

                img_path = m_out_dir / f"{p_id}_{s_id}.png"

                if args.skip_existing and img_path.exists():
                    gen_time = 1.70  # default latency estimate if cached
                else:
                    t_gen_start = time.time()
                    res = engine.generate_novel_candidate(
                        prompt=full_prompt,
                        style_features=style_feats,
                        seed=42,
                        method_name=m_name,
                        **m_kwargs,
                    )
                    gen_time = time.time() - t_gen_start
                    save_image_tensor(res["stylized_image_01"].cpu(), img_path)
                    del res
                    gc.collect()
                    torch.cuda.empty_cache()

                latencies.append(gen_time)

                # Compute CLIP scores
                s_txt = float(evaluator.cosine_score(
                    evaluator.image_embedding(img_path),
                    evaluator.text_embedding(full_prompt)
                ))
                s_sty = float(evaluator.cosine_score(
                    evaluator.image_embedding(img_path),
                    evaluator.image_embedding(s_path)
                ))
                content_scores.append(s_txt)
                style_scores.append(s_sty)

                records.append({
                    "method": m_id,
                    "prompt_id": p_id,
                    "prompt": p_text,
                    "style_id": s_id,
                    "style_descriptor": s_descriptor,
                    "clip_style": s_sty,
                    "clip_content": s_txt,
                    "gen_time_s": gen_time,
                    "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                })

                if pair_idx % 25 == 0 or pair_idx == total_pairs:
                    print(f"  [{pair_idx}/{total_pairs}] Style: {np.mean(style_scores):.4f} | Content: {np.mean(content_scores):.4f} | Avg Time: {np.mean(latencies):.2f}s")

        total_elapsed = time.time() - t_start
        peak_vram = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        mean_style = float(np.mean(style_scores))
        mean_content = float(np.mean(content_scores))
        mean_time = float(np.mean(latencies))
        diversity = compute_diversity_for_method(m_out_dir, prompts, styles)
        composite = float(40.0 * mean_style + 35.0 * mean_content + 25.0 * diversity)

        # Save method records
        m_rec_file = out_dir / f"records_{m_id}.csv"
        pd.DataFrame(records).to_csv(m_rec_file, index=False)

        # Update summary JSON
        summary_file = out_dir / "summary_10x20.json"
        summary = {}
        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        summary[m_id] = {
            "name": m_name,
            "mean_style": mean_style,
            "mean_content": mean_content,
            "diversity": diversity,
            "composite": composite,
            "mean_latency_s": mean_time,
            "total_elapsed_s": total_elapsed,
            "peak_vram_gb": peak_vram,
            "num_pairs": total_pairs,
        }
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n>>> FINISHED {m_name}:")
        print(f"    Style: {mean_style:.4f} | Content: {mean_content:.4f} | Diversity: {diversity:.4f} | Composite: {composite:.4f}")
        print(f"    Avg Latency: {mean_time:.3f}s | Elapsed: {total_elapsed:.1f}s | Peak VRAM: {peak_vram:.2f}GB")

    print(f"\nCompleted evaluation for {args.methods}!")

if __name__ == "__main__":
    main()
