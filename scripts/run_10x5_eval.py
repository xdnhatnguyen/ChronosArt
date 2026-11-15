#!/usr/bin/env python3
"""Run 10 prompts x 5 styles evaluation with PFB+SAC on Infinity-2B."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_content_ortho_batch as runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-csv", type=Path, default=PROJECT_ROOT / "prompts/content_prompts_190.csv")
    parser.add_argument("--styles-csv", type=Path, default=PROJECT_ROOT / "styles/styles_21.csv")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/grid_10x5")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-styles", type=int, default=5)
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--limit", type=int, default=None, help="Generate at most N cases (smoke testing)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(prompts_csv: Path, styles_csv: Path, num_prompts: int, num_styles: int):
    with prompts_csv.open(newline="", encoding="utf-8-sig") as f:
        prompts = list(csv.DictReader(f))[:num_prompts]
    with styles_csv.open(newline="", encoding="utf-8-sig") as f:
        styles = list(csv.DictReader(f))[:num_styles]

    cases = []
    for p in prompts:
        for s in styles:
            case_id = f"{p['content_id']}__{s['style_id']}"
            descriptor = s["style_descriptor"].strip()
            suffix = descriptor if descriptor.lower().endswith("style") else f"{descriptor} style"
            prompt_text = f"{p['content_prompt']}, {p['superclass']}, in {suffix}"
            cases.append({
                "case_id": case_id,
                "content_id": p["content_id"],
                "content_prompt": p["content_prompt"],
                "superclass": p["superclass"],
                "category": p["category"],
                "style_id": s["style_id"],
                "style_name": s["style_name"],
                "style_image": s["style_reference_image"],
                "resolved_style_path": str((PROJECT_ROOT / s["style_reference_image"]).resolve()),
                "prompt": prompt_text,
            })
    return prompts, styles, cases


def tensor_to_pil(image):
    array = image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return runtime.Image.fromarray(array, mode="RGB")


def main() -> None:
    args = parse_args()
    prompts, styles, all_cases = load_cases(args.prompts_csv, args.styles_csv, args.num_prompts, args.num_styles)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save styles for self-contained viewer
    styles_dir = output_dir / "styles"
    styles_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for s in styles:
        src = PROJECT_ROOT / s["style_reference_image"]
        dst = styles_dir / f"{s['style_id']}.png"
        if src.is_file() and not dst.exists():
            shutil.copyfile(src, dst)

    # Filter shard cases
    assigned_cases = [c for idx, c in enumerate(all_cases) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        assigned_cases = assigned_cases[:args.limit]

    print(f"Shard {args.shard_index}/{args.num_shards}: {len(assigned_cases)} cases assigned out of {len(all_cases)} total")

    # Save cases manifest for web viewer
    manifest_path = output_dir / "cases.json"
    manifest = {
        "prompts": prompts,
        "styles": styles,
        "cases": all_cases,
        "config": {
            "pn": args.pn,
            "seed": args.seed,
            "cfg": args.cfg,
            "tau": args.tau,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    stylized_dir = output_dir / "stylized"
    content_dir = output_dir / "content"
    comparison_dir = output_dir / "comparison"
    metadata_dir = output_dir / "metadata"
    for d in (stylized_dir, content_dir, comparison_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Check pending
    pending = []
    for c in assigned_cases:
        stylized_file = stylized_dir / f"{c['case_id']}.png"
        comparison_file = comparison_dir / f"{c['case_id']}.png"
        if args.overwrite or not (stylized_file.exists() and comparison_file.exists()):
            pending.append(c)

    print(f"Pending cases to run on this shard: {len(pending)} / {len(assigned_cases)}")
    if not pending:
        print("All assigned cases already complete.")
        return

    # Load runtime and model
    print("Loading Infinity runtime and models...")
    runtime.load_runtime_modules()
    import torch
    from chronosart.config import ExperimentConfig
    from chronosart.style_transfer import StyleTransferEngine

    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        paper_alpha=1.0,
        paper_pfb_feature_index=2,
        paper_sac_prediction_start=2,
        base_style_strength=1.0,
    )
    engine = StyleTransferEngine(bundle, config)
    print("Engine ready. Pre-caching style features...")
    for s in styles:
        resolved = str((PROJECT_ROOT / s["style_reference_image"]).resolve())
        engine.get_style_features(resolved)
    print("Style features pre-cached.")

    # Run generation
    start_total = time.time()
    for i, c in enumerate(pending, start=1):
        case_id = c["case_id"]
        prompt = c["prompt"]
        style_path = c["resolved_style_path"]
        print(f"[{i}/{len(pending)}] Generating {case_id}: \"{prompt}\"...")
        t0 = time.time()

        result = engine.paper_dual_path_generate(
            prompt,
            engine.get_style_features(style_path),
            pfb_feature_indices=[2],
            rank=None,
            alpha=1.0,
            style_strength=1.0,
            style_decay=1.0,
            sac_prediction_start=2,
            sac_strength=1.0,
            enable_sac=True,
            seed=args.seed,
        )
        elapsed = time.time() - t0

        stylized_img = tensor_to_pil(result["stylized_image_01"][0])
        content_img = tensor_to_pil(result["content_image_01"][0])

        stylized_path = stylized_dir / f"{case_id}.png"
        content_path = content_dir / f"{case_id}.png"
        comparison_path = comparison_dir / f"{case_id}.png"
        meta_path = metadata_dir / f"{case_id}.json"

        runtime.save_image_atomic(stylized_img, stylized_path)
        runtime.save_image_atomic(content_img, content_path)
        runtime.save_comparison(content_img, style_path, stylized_img, prompt, comparison_path)

        meta = {
            "case_id": case_id,
            "prompt": prompt,
            "content_prompt": c["content_prompt"],
            "style_name": c["style_name"],
            "elapsed_seconds": round(elapsed, 2),
            "sac_calls": result["sac_calls"],
            "max_q_copy_error": float(result["max_q_copy_error"]),
            "max_k_copy_error": float(result["max_k_copy_error"]),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Done {case_id} in {elapsed:.1f}s (avg {(time.time() - start_total) / i:.1f}s/img)")

    total_time = time.time() - start_total
    print(f"Shard {args.shard_index} finished {len(pending)} cases in {total_time:.1f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
