#!/usr/bin/env python3
"""Generate ZooSketch Assets:
Generates multi-seed sketch frames for wildlife animals using ChronosArt with 08_pencil_sketch.png,
then applies rembg / SAM background removal to produce transparent living sketches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for d in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(d))

import run_content_ortho_batch as runtime
from chronosart.config import ExperimentConfig
from chronosart.style_transfer import StyleTransferEngine
from chronosart.plotting import save_image_tensor

ANIMALS = [
    {
        "id": "elephant",
        "common_name": "Majestic Elephant",
        "scientific_name": "Loxodonta africana",
        "category": "Savanna Titan",
        "habitat": "African Savanna & Forest",
        "tagline": "THE GENTLE GIANT OF GRAPHITE",
        "lore": "Loài động vật có vú trên cạn lớn nhất hành tinh. Chiếc vòi linh hoạt và cặp ngà kiêu hãnh được tái hiện sống động qua từng thớ nét chì đan xen đa tỉ lệ.",
        "stats": {
            "weight": "6,000 kg",
            "speed": "40 km/h",
            "lifespan": "70 years",
            "status": "Vulnerable"
        },
        "prompt": "A majestic elephant walking forward, wildlife in pencil sketch style",
    },
    {
        "id": "lion",
        "common_name": "Noble Lion",
        "scientific_name": "Panthera leo",
        "category": "Apex Monarch",
        "habitat": "Grasslands & Plains",
        "tagline": "THE KING OF MONOCHROME PLAINS",
        "lore": "Chúa sơn lâm với bờm lông rậm rạp biểu trưng cho quyền uy tối thượng. Các nét chì cao tần khắc họa ánh mắt sắc lạnh và tư thế oai vệ không thể khuất phục.",
        "stats": {
            "weight": "190 kg",
            "speed": "80 km/h",
            "lifespan": "15 years",
            "status": "Vulnerable"
        },
        "prompt": "A noble lion with a majestic mane looking forward, wildlife in pencil sketch style",
    },
    {
        "id": "tiger",
        "common_name": "Fierce Tiger",
        "scientific_name": "Panthera tigris",
        "category": "Jungle Predator",
        "habitat": "Dense Tropical Rainforest",
        "tagline": "SHADOWS IN STRIPES & GRAPHITE",
        "lore": "Bậc thầy săn mồi trong rừng sâu với những dải vằn đặc trưng. Thuật toán ChronosArt bóc tách hoàn hảo kết cấu lông vằn tương phản cao mà không làm vỡ hình thể giải phẫu.",
        "stats": {
            "weight": "260 kg",
            "speed": "65 km/h",
            "lifespan": "20 years",
            "status": "Endangered"
        },
        "prompt": "A powerful tiger walking stealthily forward, wildlife in pencil sketch style",
    },
    {
        "id": "wolf",
        "common_name": "Lone Wolf",
        "scientific_name": "Canis lupus",
        "category": "Wilderness Phantom",
        "habitat": "Tundra & Boreal Forest",
        "tagline": "SPIRIT OF THE NORTHERN TUNDRA",
        "lore": "Biểu tượng bất khuất của vùng hoang dã băng giá. Lớp lông dày chịu bão tuyết được phác họa tinh xảo bằng phương pháp SVD đa tần số.",
        "stats": {
            "weight": "45 kg",
            "speed": "60 km/h",
            "lifespan": "13 years",
            "status": "Least Concern"
        },
        "prompt": "A majestic lone wolf standing proudly, wildlife in pencil sketch style",
    },
    {
        "id": "eagle",
        "common_name": "Soaring Eagle",
        "scientific_name": "Aquila chrysaetos",
        "category": "Sky Sovereign",
        "habitat": "Mountain Heights & Cliffs",
        "tagline": "THE SOVEREIGN OF THE AIR CURRENTS",
        "lore": "Sải cánh rộng lớn chao liệng giữa tầng không. Từng phiến lông vũ cánh và móng vuốt sắc nhọn được phân tầng sắc nét qua các scale mịn của kiến trúc Infinity.",
        "stats": {
            "wingspan": "2.3 m",
            "dive_speed": "320 km/h",
            "lifespan": "30 years",
            "status": "Least Concern"
        },
        "prompt": "A powerful eagle with wings spread wide, wildlife in pencil sketch style",
    },
]

SEEDS = [42, 100, 777]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style-image", type=str, default="styles/08_pencil_sketch.png")
    parser.add_argument("--web-dir", type=Path, default=PROJECT_ROOT / "zoo_web")
    args = parser.parse_args()

    web_dir = args.web_dir.resolve()
    assets_dir = web_dir / "assets/animals"
    assets_dir.mkdir(parents=True, exist_ok=True)

    style_path = (PROJECT_ROOT / args.style_image).resolve()
    if not style_path.is_file():
        raise FileNotFoundError(f"Style image not found: {style_path}")

    # Copy style image to zoo_web/assets/
    (web_dir / "assets").mkdir(parents=True, exist_ok=True)
    shutil_dest = web_dir / "assets/style_pencil_sketch.png"
    if not shutil_dest.is_file():
        import shutil
        shutil.copy2(style_path, shutil_dest)

    print("=" * 80)
    print("ZOOSKETCH ASSET GENERATION PIPELINE")
    print(f"Animals: {len(ANIMALS)} | Seeds per animal: {len(SEEDS)} | Total images: {len(ANIMALS) * len(SEEDS)}")
    print(f"Style reference: {style_path.name}")
    print("=" * 80)

    # 1. Initialize model
    print("[1/3] Loading Infinity Model Runtime...")
    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn, cfg=args.cfg, tau=args.tau, top_k=args.top_k, top_p=args.top_p, seed=42
    )
    engine = StyleTransferEngine(bundle, config)

    # Pre-extract style features
    print("[2/3] Extracting SVD Style Features from Reference...")
    style_feats = engine.get_style_features(style_path)

    # Generate images
    generated_records = []
    print("\n[3/3] Generating Multi-Seed Living Sketches...")
    for a_idx, anim in enumerate(ANIMALS, start=1):
        aid = anim["id"]
        anim_dir = assets_dir / aid
        anim_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- [{a_idx}/{len(ANIMALS)}] Generating: {anim['common_name']} ({aid}) ---")

        frame_paths = []
        for s_idx, s in enumerate(SEEDS):
            frame_file = anim_dir / f"frame_{s_idx}.png"
            if not frame_file.is_file():
                print(f"  > Generating Frame {s_idx} (Seed {s})...")
                t0 = time.time()
                res = engine.paper_dual_path_generate(
                    anim["prompt"],
                    style_feats,
                    seed=s,
                    pfb_feature_indices=[1, 2, 3],
                    style_strength=1.4,
                    style_decay=0.85,
                    sac_prediction_start=2,
                    sac_strength=0.75,
                    sac_head_ratio=0.70,
                    use_geodesic=True,
                    use_fourier_phase=True,
                    fourier_scales=[5],
                    fourier_kappa=0.28,
                    use_scale_scheduled_tau=True,
                    tau_coarse=0.15,
                    tau_fine_min=0.85,
                    tau_fine_max=1.25,
                    top_p_fine=0.92,
                    use_style_token_jitter=True,
                    jitter_start_step=5,
                    jitter_dropout=0.15,
                )
                save_image_tensor(res["stylized_image_01"].cpu(), frame_file)
                print(f"    Done in {time.time() - t0:.2f}s -> {frame_file.name}")
            else:
                print(f"  > Frame {s_idx} (Seed {s}) already exists.")

            frame_paths.append(str(frame_file))

        anim_copy = dict(anim)
        anim_copy["frame_files"] = [f"assets/animals/{aid}/frame_{i}.png" for i in range(len(SEEDS))]
        anim_copy["nobg_files"] = [f"assets/animals/{aid}/frame_{i}_nobg.png" for i in range(len(SEEDS))]
        generated_records.append(anim_copy)

    # Save JSON database
    data_file = web_dir / "data/animals.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(generated_records, f, ensure_ascii=False, indent=2)
    print(f"\nMetadata saved to {data_file}")

    # 4. Trigger Background Removal using rbm environment
    print("\n" + "=" * 80)
    print("STARTING BACKGROUND REMOVAL (TRANSPARENT PNG EXTRACTION VIA REMBG / SAM)")
    print("=" * 80)

    bg_script = f"""
from rembg import remove
from PIL import Image
from pathlib import Path

base_dir = Path('{assets_dir}')
for anim_dir in base_dir.iterdir():
    if not anim_dir.is_dir():
        continue
    for f in anim_dir.glob('frame_*.png'):
        if f.name.endswith('_nobg.png'):
            continue
        out_f = anim_dir / f.name.replace('.png', '_nobg.png')
        if not out_f.is_file():
            print(f'Removing background for {{f}}...')
            im = Image.open(f)
            nobg = remove(im)
            nobg.save(out_f, 'PNG')
            print(f'Saved -> {{out_f.name}}')
print('Background removal completed successfully!')
"""
    bg_runner = PROJECT_ROOT / "scripts/run_rembg.py"
    bg_runner.write_text(bg_script, encoding="utf-8")

    cmd = ["/home/khoan/.conda/envs/rbm/bin/python", str(bg_runner)]
    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    print(ret.stdout)
    if ret.stderr:
        print("Stderr / warnings:", ret.stderr)

    print("\nALL ASSETS GENERATED & PROCESSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
