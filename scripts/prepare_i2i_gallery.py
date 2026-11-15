"""Prepare gallery data and baseline images for ChronosArt-I2I web viewer."""
import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third-party" / "Infinity"))
sys.path.insert(0, str(PROJECT_ROOT / "third-party" / "Infinity" / "tools"))

from chronosart.infinity_runtime import load_bundle
from chronosart.svd_blending import truncated_svd

def generate_baseline_svd(c_path, s_path, bundle, svd_rank=8, svd_str=0.75):
    c_img = Image.open(c_path).convert("RGB").resize((512, 512))
    s_img = Image.open(s_path).convert("RGB").resize((512, 512))

    c_m11 = TF.to_tensor(c_img).unsqueeze(0).cuda().mul(2).sub(1)
    s_m11 = TF.to_tensor(s_img).unsqueeze(0).cuda().mul(2).sub(1)

    scale_schedule = bundle.scale_schedule
    final_size = scale_schedule[-1]
    num_scales = len(scale_schedule)

    with torch.no_grad():
        _, _, _, c_bits, _, _ = bundle.vae.encode(c_m11.float(), scale_schedule=scale_schedule)
        _, _, _, s_bits, _, _ = bundle.vae.encode(s_m11.float(), scale_schedule=scale_schedule)

    summed_codes = None
    for s in range(num_scales):
        cc = bundle.vae.quantizer.lfq.indices_to_codes(c_bits[s], label_type="bit_label")
        sc = bundle.vae.quantizer.lfq.indices_to_codes(s_bits[s], label_type="bit_label")
        if s >= 4:
            c_f = cc.squeeze(2) if cc.ndim == 5 else cc
            s_f = sc.squeeze(2) if sc.ndim == 5 else sc
            delta = truncated_svd(s_f, rank=svd_rank, alpha=0.0) - truncated_svd(c_f, rank=svd_rank, alpha=0.0)
            code = c_f + float(svd_str) * delta
            if cc.ndim == 5:
                code = code.unsqueeze(2)
        else:
            code = cc
        if s != num_scales - 1:
            code = F.interpolate(code, size=final_size, mode=bundle.vae.quantizer.z_interplote_up)
        summed_codes = code if summed_codes is None else summed_codes + code

    decoded = bundle.vae.decode(summed_codes.squeeze(-3))
    out_img = decoded.add(1).mul(0.5).clamp(0, 1)
    return out_img

def main():
    eval_csv_path = PROJECT_ROOT / "auto_research_i2i" / ".eval_tmp" / "chronosart_i2i_eval.csv"
    if not eval_csv_path.exists():
        print("Error: Eval CSV not found!")
        return

    df = pd.read_csv(eval_csv_path)
    df = df[df['content_name'] != 'AVERAGE'].copy()
    print(f"Loaded {len(df)} evaluated pairs from CSV.")

    out_dir = PROJECT_ROOT / "auto_research_i2i" / ".eval_tmp"
    
    # Check how many baselines need generating
    missing_baselines = []
    for _, row in df.iterrows():
        b_name = f"baseline_{row['content_name']}_{row['style_name']}.png"
        b_path = out_dir / b_name
        if not b_path.exists():
            missing_baselines.append((row['content_name'], row['style_name'], b_path))

    print(f"Missing baseline images: {len(missing_baselines)} / {len(df)}")
    if missing_baselines:
        print("Generating missing baseline images on GPU...")
        args = argparse.Namespace(pn="0.25M", seed=42)
        bundle = load_bundle(args)
        t0 = time.time()
        for idx, (c_name, s_name, b_path) in enumerate(missing_baselines):
            c_path = PROJECT_ROOT / "content" / f"{c_name}.png"
            s_path = PROJECT_ROOT / "styles" / f"{s_name}.png"
            base_t = generate_baseline_svd(c_path, s_path, bundle)
            TF.to_pil_image(base_t[0].cpu()).save(b_path)
            if (idx + 1) % 25 == 0 or idx == len(missing_baselines) - 1:
                print(f"  Generated {idx + 1}/{len(missing_baselines)} baselines ({time.time() - t0:.1f}s)...")
        print("All baseline images ready!")

    # Build web gallery JSON
    web_dir = PROJECT_ROOT / "i2i_viewer"
    data_dir = web_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for idx, row in df.iterrows():
        c_name = row['content_name']
        s_name = row['style_name']
        
        csd = float(row['csd_style_similarity']) * 100.0
        dino = float(row['dino_content_similarity']) * 100.0
        clip_i = float(row['clip_i_content_similarity']) * 100.0
        lpips = float(row['lpips']) * 100.0
        dcl = float(row['dcl']) * 100.0
        
        # Harmonic Balance Score
        s_style = max(0.0, csd)
        s_content = max(0.0, clip_i - 0.5 * lpips - 0.5 * dcl)
        harmonic = (2.0 * s_style * s_content) / (s_style + s_content + 1e-6)

        pair_item = {
            "id": f"{c_name}_{s_name}",
            "index": idx,
            "content_name": c_name,
            "content_title": c_name.replace("_", " ").title(),
            "style_name": s_name,
            "style_title": s_name.replace("style_", "").replace("_", " ").title(),
            "content_img": f"/images/content/{c_name}.png",
            "style_img": f"/images/styles/{s_name}.png",
            "baseline_img": f"/images/eval/baseline_{c_name}_{s_name}.png",
            "ours_img": f"/images/eval/chronosart_i2i_{c_name}_{s_name}.png",
            "metrics": {
                "csd": round(csd, 2),
                "dino": round(dino, 2),
                "clip_i": round(clip_i, 2),
                "lpips": round(lpips, 2),
                "dcl": round(dcl, 2),
                "harmonic": round(harmonic, 2),
                "retrieval_correct": bool(row.get('style_retrieval_correct', 1.0) == 1.0),
            }
        }
        pairs.append(pair_item)

    # Sort default by harmonic descending
    pairs_sorted = sorted(pairs, key=lambda x: x["metrics"]["harmonic"], reverse=True)

    json_path = data_dir / "pairs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pairs_sorted, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {json_path} with {len(pairs)} pairs!")

if __name__ == "__main__":
    main()
