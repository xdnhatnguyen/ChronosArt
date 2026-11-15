"""auto_research/evaluate.py - The Locked Judge.

In the Karpathy AutoResearch protocol, this file is STRICTLY READ-ONLY.
The AI Agent is forbidden from modifying this file to avoid gaming the benchmark.

Fixed Protocol:
- Benchmark: 4 canonical style-content test cases, 2 seeds each (8 images total)
- Fixed Wall-Clock Budget: Target ~2-3 minutes total runtime on a single GPU
- Objective Function (Single Number, Higher is Better):
    composite_score = 40.0 * clip_style + 35.0 * clip_content + 25.0 * diversity
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
infinity_dir = PROJECT_ROOT / "third-party" / "Infinity" if (PROJECT_ROOT / "third-party" / "Infinity").exists() else PROJECT_ROOT / "Infinity"
for d in (PROJECT_ROOT / "src", infinity_dir, infinity_dir / "tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(d))

import run_content_ortho_batch as runtime
from chronosart.config import ExperimentConfig
from chronosart.style_transfer import StyleTransferEngine
from chronosart.clip_metrics import CLIPMetricsEvaluator
from chronosart.plotting import save_image_tensor

# Import candidate logic from candidate.py
from auto_research.candidate import get_inference_args, CONFIG

# =============================================================================
# FIXED BENCHMARK SUITE (4 diverse styles, 4 distinct objects)
# =============================================================================
BENCHMARK_CASES = [
    {
        "id": "case1_cat_glowing",
        "prompt": "a cute cat sitting peacefully on a cushion",
        "style_image": "styles/style_10_glowing_neon.png",
    },
    {
        "id": "case2_kettle_origami",
        "prompt": "a kettle sitting on a wooden table",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case3_bear_crayon",
        "prompt": "a wild bear standing in the deep forest",
        "style_image": "styles/style_04_kid_crayon_drawing.png",
    },
    {
        "id": "case4_owl_sketch",
        "prompt": "a wise owl perched on a tree branch",
        "style_image": "styles/08_pencil_sketch.png",
    },
]

SEEDS = [42, 108]  # 2 seeds per case = 8 inferences total (~2 minutes)


def compute_diversity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Mean absolute pixel difference between two seeds, normalized to [0, 1]."""
    diff = np.abs(img_a.astype(float) - img_b.astype(float)).mean()
    return float(diff / 255.0)


def evaluate() -> dict:
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    # Setup temporary directory for current run
    temp_dir = PROJECT_ROOT / "auto_research" / ".eval_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Engine & CLIP Evaluator
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    args, _ = parser.parse_known_args()

    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn, cfg=args.cfg, tau=args.tau, top_k=args.top_k, top_p=args.top_p, seed=42
    )
    engine = StyleTransferEngine(bundle, config)
    evaluator = CLIPMetricsEvaluator(output_dir=temp_dir)

    all_style_scores = []
    all_content_scores = []
    all_diversity_scores = []
    image_paths_to_cleanup = []

    for case in BENCHMARK_CASES:
        style_path = (PROJECT_ROOT / case["style_image"]).resolve()
        if not style_path.exists():
            raise FileNotFoundError(f"Missing style file: {style_path}")

        style_feats = engine.get_style_features(style_path)
        case_imgs = []

        for seed in SEEDS:
            out_file = temp_dir / f"{case['id']}_seed{seed}.png"
            image_paths_to_cleanup.append(out_file)

            # Retrieve candidate arguments
            kwargs = get_inference_args(seed=seed)

            # Execute generation
            res = engine.paper_dual_path_generate(
                prompt=case["prompt"],
                style_features=style_feats,
                **kwargs,
            )
            save_image_tensor(res["stylized_image_01"].cpu(), out_file)
            
            # Metrics
            img_np = np.array(Image.open(out_file))
            case_imgs.append(img_np)

            s_txt = evaluator.cosine_score(
                evaluator.image_embedding(out_file), evaluator.text_embedding(case["prompt"])
            )
            s_sty = evaluator.cosine_score(
                evaluator.image_embedding(out_file), evaluator.image_embedding(style_path)
            )
            all_content_scores.append(float(s_txt))
            all_style_scores.append(float(s_sty))

            del res
            gc.collect()
            torch.cuda.empty_cache()

        # Compute diversity between seed 42 and seed 108
        div = compute_diversity(case_imgs[0], case_imgs[1])
        all_diversity_scores.append(div)

        del style_feats
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate metrics
    mean_style = float(np.mean(all_style_scores))
    mean_content = float(np.mean(all_content_scores))
    mean_diversity = float(np.mean(all_diversity_scores))

    # Single Composite Objective Function:
    # 40% Style Fidelity + 35% Content Semantics + 25% Seed Diversity
    composite_score = float(40.0 * mean_style + 35.0 * mean_content + 25.0 * mean_diversity)

    elapsed_sec = float(time.time() - t_start)
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    # Clean temporary files
    for p in image_paths_to_cleanup:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    return {
        "composite_score": composite_score,
        "clip_style": mean_style,
        "clip_content": mean_content,
        "diversity": mean_diversity,
        "eval_seconds": elapsed_sec,
        "peak_vram_mb": peak_vram_mb,
        "num_images": len(BENCHMARK_CASES) * len(SEEDS),
        "status": "SUCCESS",
    }


def main():
    try:
        results = evaluate()
        print("\n---")
        print(f"composite_score:  {results['composite_score']:.4f}")
        print(f"clip_style:       {results['clip_style']:.4f}")
        print(f"clip_content:     {results['clip_content']:.4f}")
        print(f"diversity:        {results['diversity']:.4f}")
        print(f"eval_seconds:     {results['eval_seconds']:.1f}")
        print(f"peak_vram_mb:     {results['peak_vram_mb']:.1f}")
        print(f"num_images:       {results['num_images']}")
        print(f"status:           {results['status']}")
        
        # Save last evaluation record
        record_file = PROJECT_ROOT / "auto_research" / ".last_eval.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        sys.exit(0)
    except Exception as exc:
        print("\n---")
        print("composite_score:  0.0000")
        print("status:           CRASH")
        print(f"error:            {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
