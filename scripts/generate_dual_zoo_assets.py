#!/usr/bin/env python3
"""Generate Dual Zoo Assets:
1. SketchZoo (08_pencil_sketch.png) - 10 Animals x 3 Seeds
2. OrigamiZoo (style_01_colorful_origami.png) - 10 Animals x 3 Seeds
Then converts all frames to optimized WebP format with SAM/rembg transparent backgrounds.
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
        "lore": "Loài thú trên cạn khổng lồ với đôi tai lớn và chiếc vòi khéo léo. Cấu trúc cơ bắp vĩ mô được khóa chặt vững chãi.",
        "stats": { "weight": "6,000 kg", "speed": "40 km/h", "lifespan": "70 yrs" },
        "sketch_prompt": "A majestic elephant walking forward, wildlife in pencil sketch style",
        "origami_prompt": "A majestic elephant walking forward, wildlife in colorful origami style",
    },
    {
        "id": "lion",
        "common_name": "Noble Lion",
        "scientific_name": "Panthera leo",
        "category": "Apex Monarch",
        "lore": "Chúa tể thảo nguyên với bờm lông quyền uy. Các nếp gấp góc cạnh hoặc thớ chì vi mô làm nổi bật thần thái oai nghiêm.",
        "stats": { "weight": "190 kg", "speed": "80 km/h", "lifespan": "15 yrs" },
        "sketch_prompt": "A noble lion with a majestic mane looking forward, wildlife in pencil sketch style",
        "origami_prompt": "A noble lion with a majestic mane looking forward, wildlife in colorful origami style",
    },
    {
        "id": "tiger",
        "common_name": "Fierce Tiger",
        "scientific_name": "Panthera tigris",
        "category": "Jungle Predator",
        "lore": "Bậc thầy ngụy trang với những dải vằn tương phản. Thuật toán ChronosArt bảo toàn hoàn hảo các đường vân sọc đặc trưng.",
        "stats": { "weight": "260 kg", "speed": "65 km/h", "lifespan": "20 yrs" },
        "sketch_prompt": "A powerful tiger walking stealthily forward, wildlife in pencil sketch style",
        "origami_prompt": "A powerful tiger walking stealthily forward, wildlife in colorful origami style",
    },
    {
        "id": "wolf",
        "common_name": "Lone Wolf",
        "scientific_name": "Canis lupus",
        "category": "Wilderness Phantom",
        "lore": "Chiến binh bất khuất của vùng cực giá lạnh. Lớp lông rậm rạp được tái tạo qua các nếp gấp đa giác sắc sảo.",
        "stats": { "weight": "45 kg", "speed": "60 km/h", "lifespan": "13 yrs" },
        "sketch_prompt": "A majestic lone wolf standing proudly, wildlife in pencil sketch style",
        "origami_prompt": "A majestic lone wolf standing proudly, wildlife in colorful origami style",
    },
    {
        "id": "eagle",
        "common_name": "Soaring Eagle",
        "scientific_name": "Aquila chrysaetos",
        "category": "Sky Sovereign",
        "lore": "Chúa tể bầu trời với sải cánh uy lực. Từng phiến lông vũ được phân tầng hình học ấn tượng.",
        "stats": { "wingspan": "2.3 m", "dive_speed": "320 km/h", "lifespan": "30 yrs" },
        "sketch_prompt": "A powerful eagle with wings spread wide, wildlife in pencil sketch style",
        "origami_prompt": "A powerful eagle with wings spread wide, wildlife in colorful origami style",
    },
    {
        "id": "deer",
        "common_name": "Graceful Deer",
        "scientific_name": "Cervus elaphus",
        "category": "Forest Spirit",
        "lore": "Sinh linh thanh thoát của rừng sâu với cặp gạc phân nhánh kiêu hãnh và dáng đứng nhẹ nhàng uyển chuyển.",
        "stats": { "weight": "160 kg", "speed": "75 km/h", "lifespan": "18 yrs" },
        "sketch_prompt": "A graceful deer with branching antlers standing in forest, wildlife in pencil sketch style",
        "origami_prompt": "A graceful deer with branching antlers standing in forest, wildlife in colorful origami style",
    },
    {
        "id": "fox",
        "common_name": "Red Fox",
        "scientific_name": "Vulpes vulpes",
        "category": "Shadow Trickster",
        "lore": "Loài cáo thông minh và nhanh nhẹn với chiếc đuôi bông bềnh rực rỡ, thoắt ẩn thoắt hiện trong thung lũng sương mù.",
        "stats": { "weight": "8 kg", "speed": "50 km/h", "lifespan": "10 yrs" },
        "sketch_prompt": "A clever red fox with a bushy tail sitting, wildlife in pencil sketch style",
        "origami_prompt": "A clever red fox with a bushy tail sitting, wildlife in colorful origami style",
    },
    {
        "id": "panda",
        "common_name": "Giant Panda",
        "scientific_name": "Ailuropoda melanoleuca",
        "category": "Bamboo Sage",
        "lore": "Biểu tượng hòa bình và vẻ đẹp đáng yêu của rừng trúc phương Đông với hai mảng khoang đen trắng kinh điển.",
        "stats": { "weight": "120 kg", "speed": "32 km/h", "lifespan": "20 yrs" },
        "sketch_prompt": "A cute giant panda sitting and eating bamboo, wildlife in pencil sketch style",
        "origami_prompt": "A cute giant panda sitting and eating bamboo, wildlife in colorful origami style",
    },
    {
        "id": "owl",
        "common_name": "Wise Owl",
        "scientific_name": "Bubo virginianus",
        "category": "Nocturnal Oracle",
        "lore": "Kẻ thống trị màn đêm với đôi mắt tròn sáng quắc và khả năng định vị con mồi trong bóng tối tuyệt đối.",
        "stats": { "wingspan": "1.5 m", "speed": "65 km/h", "lifespan": "15 yrs" },
        "sketch_prompt": "A majestic horned owl perched on a branch, wildlife in pencil sketch style",
        "origami_prompt": "A majestic horned owl perched on a branch, wildlife in colorful origami style",
    },
    {
        "id": "chameleon",
        "common_name": "Prism Chameleon",
        "scientific_name": "Chamaeleo calyptratus",
        "category": "Chromatic Mystic",
        "lore": "Bậc thầy biến hóa màu sắc với chiếc đuôi cuộn xoắn và cặp mắt độc lập xoay chuyển 360 độ.",
        "stats": { "length": "45 cm", "tongue_reach": "1.5x body", "lifespan": "7 yrs" },
        "sketch_prompt": "A colorful chameleon perched on a twig, wildlife in pencil sketch style",
        "origami_prompt": "A colorful chameleon perched on a twig, wildlife in colorful origami style",
    },
]

SEEDS = [42, 100, 777]


def convert_to_webp(src_path: Path, dest_path: Path, quality: int = 90):
    im = Image.open(src_path)
    im.save(dest_path, "WEBP", quality=quality, method=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--web-dir", type=Path, default=PROJECT_ROOT / "zoo_web")
    args = parser.parse_args()

    web_dir = args.web_dir.resolve()
    assets_dir = web_dir / "assets/animals"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sketch_style_path = (PROJECT_ROOT / "styles/08_pencil_sketch.png").resolve()
    origami_style_path = (PROJECT_ROOT / "styles/style_01_colorful_origami.png").resolve()

    print("=" * 80)
    print("DUAL ZOOSKETCH ASSET GENERATION PIPELINE")
    print(f"Animals: {len(ANIMALS)} | Seeds: {len(SEEDS)} | Styles: 2 (Sketch + Origami)")
    print("=" * 80)

    # 1. Initialize model
    print("[1/4] Loading Infinity Runtime...")
    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn, cfg=args.cfg, tau=args.tau, top_k=args.top_k, top_p=args.top_p, seed=42
    )
    engine = StyleTransferEngine(bundle, config)

    # Pre-extract style features
    print("[2/4] Extracting SVD Style Features...")
    sketch_feats = engine.get_style_features(sketch_style_path)
    origami_feats = engine.get_style_features(origami_style_path)

    # 2. Generation Loop
    print("\n[3/4] Generating Multi-Seed Frames for both Styles...")
    styles_info = [
        ("sketch", sketch_feats, "sketch_prompt"),
        ("origami", origami_feats, "origami_prompt"),
    ]

    for anim in ANIMALS:
        aid = anim["id"]
        anim_dir = assets_dir / aid
        anim_dir.mkdir(parents=True, exist_ok=True)

        for style_name, feats, prompt_key in styles_info:
            prompt = anim[prompt_key]
            for s_idx, s in enumerate(SEEDS):
                frame_png = anim_dir / f"{style_name}_frame_{s_idx}.png"
                if not frame_png.is_file():
                    print(f"  > [{aid.upper()}] Style: {style_name} | Frame {s_idx} (Seed {s})...")
                    t0 = time.time()
                    res = engine.paper_dual_path_generate(
                        prompt,
                        feats,
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
                    save_image_tensor(res["stylized_image_01"].cpu(), frame_png)
                    print(f"    Done in {time.time() - t0:.2f}s -> {frame_png.name}")
                else:
                    print(f"  > [{aid.upper()}] {style_name}_frame_{s_idx}.png exists.")

    print("\n[4/4] Removing Backgrounds via Rembg and converting to optimized WebP...")
    bg_script = f"""
from rembg import remove
from PIL import Image
from pathlib import Path

base_dir = Path('{assets_dir}')
for anim_dir in base_dir.iterdir():
    if not anim_dir.is_dir():
        continue
    for f in anim_dir.glob('*_frame_*.png'):
        if f.name.endswith('_nobg.png'):
            continue
        nobg_png = anim_dir / f.name.replace('.png', '_nobg.png')
        if not nobg_png.is_file():
            print(f'Removing BG: {{f.name}}...')
            im = Image.open(f)
            out = remove(im)
            out.save(nobg_png, 'PNG')

        # Convert both to WebP
        webp_full = anim_dir / f.name.replace('.png', '.webp')
        if not webp_full.is_file():
            Image.open(f).save(webp_full, 'WEBP', quality=90, method=6)

        webp_nobg = anim_dir / f.name.replace('.png', '_nobg.webp')
        if not webp_nobg.is_file() and nobg_png.is_file():
            Image.open(nobg_png).save(webp_nobg, 'WEBP', quality=90, method=6)

print('Background removal and WebP conversion complete!')
"""
    bg_runner = PROJECT_ROOT / "scripts/run_dual_rembg.py"
    bg_runner.write_text(bg_script, encoding="utf-8")

    cmd = ["/home/khoan/.conda/envs/rbm/bin/python", str(bg_runner)]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    print(ret.stdout)
    if ret.stderr:
        print("Stderr:", ret.stderr)

    # 3. Update metadata JSON
    final_records = []
    for anim in ANIMALS:
        aid = anim["id"]
        record = dict(anim)
        record["sketch"] = {
            "prompt": anim["sketch_prompt"],
            "full_webp": [f"assets/animals/{aid}/sketch_frame_{i}.webp" for i in range(len(SEEDS))],
            "nobg_webp": [f"assets/animals/{aid}/sketch_frame_{i}_nobg.webp" for i in range(len(SEEDS))],
        }
        record["origami"] = {
            "prompt": anim["origami_prompt"],
            "full_webp": [f"assets/animals/{aid}/origami_frame_{i}.webp" for i in range(len(SEEDS))],
            "nobg_webp": [f"assets/animals/{aid}/origami_frame_{i}_nobg.webp" for i in range(len(SEEDS))],
        }
        final_records.append(record)

    json_path = web_dir / "data/animals_dual.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    print(f"\nFinal Dual-Zoo database written to: {json_path}")


if __name__ == "__main__":
    main()
