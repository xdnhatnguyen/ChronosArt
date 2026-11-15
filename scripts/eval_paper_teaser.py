#!/usr/bin/env python3
"""Run and visualize the 18 cases (3 styles x 6 prompts) from teaser6_mod.pdf."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
infinity_dir = PROJECT_ROOT / "third-party" / "Infinity" if (PROJECT_ROOT / "third-party" / "Infinity").exists() else PROJECT_ROOT / "Infinity"
for directory in (PROJECT_ROOT / "src", infinity_dir, infinity_dir / "tools", PROJECT_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_content_ortho_batch as runtime

TEASER_CASES = [
    # Row 1: Style 01 (Colorful Origami)
    {
        "row": 1,
        "col": 1,
        "style_id": "style_01",
        "style_name": "Colorful Origami",
        "style_descriptor": "colorful origami",
        "style_image": "styles/style_01_colorful_origami.png",
        "content_prompt": "A kettle",
        "prompt": "A kettle in colorful origami style",
    },
    {
        "row": 1,
        "col": 2,
        "style_id": "style_01",
        "style_name": "Colorful Origami",
        "style_descriptor": "colorful origami",
        "style_image": "styles/style_01_colorful_origami.png",
        "content_prompt": "An armchair",
        "prompt": "An armchair in colorful origami style",
    },
    {
        "row": 1,
        "col": 3,
        "style_id": "style_01",
        "style_name": "Colorful Origami",
        "style_descriptor": "colorful origami",
        "style_image": "styles/style_01_colorful_origami.png",
        "content_prompt": "A table",
        "prompt": "A table in colorful origami style",
    },
    {
        "row": 1,
        "col": 4,
        "style_id": "style_01",
        "style_name": "Colorful Origami",
        "style_descriptor": "colorful origami",
        "style_image": "styles/style_01_colorful_origami.png",
        "content_prompt": "A pair of shoes",
        "prompt": "A pair of shoes in colorful origami style",
    },
    {
        "row": 1,
        "col": 5,
        "style_id": "style_01",
        "style_name": "Colorful Origami",
        "style_descriptor": "colorful origami",
        "style_image": "styles/style_01_colorful_origami.png",
        "content_prompt": "A car",
        "prompt": "A car in colorful origami style",
    },
    {
        "row": 1,
        "col": 6,
        "style_id": "style_01",
        "style_name": "Colorful Origami",
        "style_descriptor": "colorful origami",
        "style_image": "styles/style_01_colorful_origami.png",
        "content_prompt": "A ball",
        "prompt": "A ball in colorful origami style",
    },
    # Row 2: Style 02 (3D Rendering / Toy House)
    {
        "row": 2,
        "col": 1,
        "style_id": "style_02",
        "style_name": "3D Rendering",
        "style_descriptor": "3d rendering",
        "style_image": "styles/style_02_3d_rendering.png",
        "content_prompt": "A pen",
        "prompt": "A pen in 3d rendering style",
    },
    {
        "row": 2,
        "col": 2,
        "style_id": "style_02",
        "style_name": "3D Rendering",
        "style_descriptor": "3d rendering",
        "style_image": "styles/style_02_3d_rendering.png",
        "content_prompt": "A coral",
        "prompt": "A coral in 3d rendering style",
    },
    {
        "row": 2,
        "col": 3,
        "style_id": "style_02",
        "style_name": "3D Rendering",
        "style_descriptor": "3d rendering",
        "style_image": "styles/style_02_3d_rendering.png",
        "content_prompt": "A bell",
        "prompt": "A bell in 3d rendering style",
    },
    {
        "row": 2,
        "col": 4,
        "style_id": "style_02",
        "style_name": "3D Rendering",
        "style_descriptor": "3d rendering",
        "style_image": "styles/style_02_3d_rendering.png",
        "content_prompt": "A helmet",
        "prompt": "A helmet in 3d rendering style",
    },
    {
        "row": 2,
        "col": 5,
        "style_id": "style_02",
        "style_name": "3D Rendering",
        "style_descriptor": "3d rendering",
        "style_image": "styles/style_02_3d_rendering.png",
        "content_prompt": "A bird",
        "prompt": "A bird in 3d rendering style",
    },
    {
        "row": 2,
        "col": 6,
        "style_id": "style_02",
        "style_name": "3D Rendering",
        "style_descriptor": "3d rendering",
        "style_image": "styles/style_02_3d_rendering.png",
        "content_prompt": "A book",
        "prompt": "A book in 3d rendering style",
    },
    # Row 3: Style 03 (Northern Renaissance Art)
    {
        "row": 3,
        "col": 1,
        "style_id": "style_03",
        "style_name": "Northern Renaissance Art",
        "style_descriptor": "northern renaissance art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
        "content_prompt": "An hourglass",
        "prompt": "An hourglass in northern renaissance art style",
    },
    {
        "row": 3,
        "col": 2,
        "style_id": "style_03",
        "style_name": "Northern Renaissance Art",
        "style_descriptor": "northern renaissance art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
        "content_prompt": "A telescope",
        "prompt": "A telescope in northern renaissance art style",
    },
    {
        "row": 3,
        "col": 3,
        "style_id": "style_03",
        "style_name": "Northern Renaissance Art",
        "style_descriptor": "northern renaissance art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
        "content_prompt": "A castle with tall spires reflected in water",
        "prompt": "A castle with tall spires reflected in water in northern renaissance art style",
    },
    {
        "row": 3,
        "col": 4,
        "style_id": "style_03",
        "style_name": "Northern Renaissance Art",
        "style_descriptor": "northern renaissance art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
        "content_prompt": "A piano",
        "prompt": "A piano in northern renaissance art style",
    },
    {
        "row": 3,
        "col": 5,
        "style_id": "style_03",
        "style_name": "Northern Renaissance Art",
        "style_descriptor": "northern renaissance art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
        "content_prompt": "A horse running in a field",
        "prompt": "A horse running in a field in northern renaissance art style",
    },
    {
        "row": 3,
        "col": 6,
        "style_id": "style_03",
        "style_name": "Northern Renaissance Art",
        "style_descriptor": "northern renaissance art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
        "content_prompt": "A train",
        "prompt": "A train in northern renaissance art style",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/teaser6")
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "0.60M", "1M"), default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--html-only", action="store_true", help="Only generate HTML viewer without inference")
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "_", text)


def tensor_to_pil(image):
    array = image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return runtime.Image.fromarray(array, mode="RGB")


def build_teaser_html(output_dir: Path, cases: list[dict]) -> str:
    # Group by row
    rows = {}
    for c in cases:
        rows.setdefault(c["row"], []).append(c)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChronosArt - Teaser Reproduction (Figure 1: teaser6_mod)</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #1f6feb;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            max-width: 1400px;
            margin: 0 auto 28px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        .header p {{
            font-size: 15px;
            color: var(--text-secondary);
            max-width: 900px;
            margin: 0 auto;
            line-height: 1.5;
        }}
        .badge {{
            display: inline-block;
            background-color: #238636;
            color: white;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 20px;
            margin-bottom: 12px;
        }}
        .grid-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        .teaser-row {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }}
        .row-title {{
            font-size: 16px;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .row-grid {{
            display: grid;
            grid-template-columns: 1.2fr repeat(6, 1fr);
            gap: 12px;
            align-items: stretch;
        }}
        .item-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            background: #0d1117;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .item-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent);
        }}
        .item-card.style-card {{
            border: 2px solid #e3b341;
            background: rgba(227, 179, 65, 0.05);
        }}
        .img-wrapper {{
            width: 100%;
            aspect-ratio: 1;
            border-radius: 6px;
            overflow: hidden;
            background: #21262d;
            margin-bottom: 8px;
            cursor: pointer;
        }}
        .img-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.3s;
        }}
        .img-wrapper:hover img {{
            transform: scale(1.05);
        }}
        .caption {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            font-style: italic;
            margin-top: 4px;
        }}
        .style-caption {{
            font-size: 13px;
            font-weight: 700;
            color: #f2cc60;
            margin-top: 4px;
        }}
        .footer {{
            max-width: 1400px;
            margin: 40px auto 20px;
            text-align: center;
            font-size: 13px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
        }}
        /* Modal for zoom */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.85);
            backdrop-filter: blur(5px);
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }}
        .modal img {{
            max-width: 80%;
            max-height: 80%;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        }}
        .modal-caption {{
            margin-top: 16px;
            font-size: 16px;
            color: var(--text-primary);
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">ChronosArt / SVD-Based Feature Decomposition</span>
        <h1>Teaser Figure Reproduction (Figure 1: teaser6_mod)</h1>
        <p>Style-personalized image generation results produced by our PFB + SAC method on Infinity-2B. Given reference style images and text prompts, the method generates images with consistent style and diverse content without training.</p>
    </div>

    <div class="grid-container">
"""

    for row_idx, row_cases in sorted(rows.items()):
        first = row_cases[0]
        style_rel = os.path.relpath(PROJECT_ROOT / first["style_image"], output_dir)
        html += f"""
        <div class="teaser-row">
            <div class="row-title">
                Row {row_idx}: {first["style_name"]}
            </div>
            <div class="row-grid">
                <div class="item-card style-card">
                    <div class="img-wrapper" onclick="openModal('{style_rel}', 'Style Reference: {first['style_name']}')">
                        <img src="{style_rel}" alt="Style Reference">
                    </div>
                    <div class="style-caption">Style Reference</div>
                </div>
"""
        for c in sorted(row_cases, key=lambda x: x["col"]):
            img_rel = c["saved_rel_path"]
            html += f"""
                <div class="item-card">
                    <div class="img-wrapper" onclick="openModal('{img_rel}', '&ldquo;{c['content_prompt']}&rdquo;')">
                        <img src="{img_rel}" alt="{c['content_prompt']}">
                    </div>
                    <div class="caption">&ldquo;{c['content_prompt']}&rdquo;</div>
                </div>
"""
        html += """
            </div>
        </div>
"""

    html += """
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <img id="modalImg" src="" alt="Zoomed image">
        <div id="modalCap" class="modal-caption"></div>
    </div>

    <div class="footer">
        Generated with ChronosArt (PFB + SAC on Infinity-2B) &bull; Inference resolution: 512&times;512 &bull; Training-Free
    </div>

    <script>
        function openModal(src, caption) {
            const modal = document.getElementById('imageModal');
            document.getElementById('modalImg').src = src;
            document.getElementById('modalCap').innerHTML = caption;
            modal.style.display = 'flex';
        }
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeModal();
        });
    </script>
</body>
</html>
"""
    return html


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        for c in TEASER_CASES:
            slug = slugify(c["content_prompt"])
            c["saved_rel_path"] = f"row{c['row']}_{c['col']:02d}_{slug}.png"
        html_content = build_teaser_html(output_dir, TEASER_CASES)
        html_file = output_dir / "index.html"
        html_file.write_text(html_content, encoding="utf-8")
        print(f"Generated HTML viewer: {html_file}")
        return

    runtime.load_runtime_modules()
    import torch
    from tqdm.auto import tqdm
    from chronosart.config import ExperimentConfig
    from chronosart.plotting import save_image_tensor
    from chronosart.style_transfer import StyleTransferEngine

    # Sharding
    sharded_cases = [c for i, c in enumerate(TEASER_CASES) if i % args.num_shards == args.shard_index]
    print(f"[Shard {args.shard_index}/{args.num_shards}] Processing {len(sharded_cases)} cases...")

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

    progress = tqdm(sharded_cases, desc=f"Shard {args.shard_index}")
    for case in progress:
        slug = slugify(case["content_prompt"])
        out_name = f"row{case['row']}_{case['col']:02d}_{slug}.png"
        out_path = output_dir / out_name
        case["saved_path"] = str(out_path)
        case["saved_rel_path"] = out_name

        if out_path.is_file() and not args.overwrite:
            progress.set_postfix_str(f"Skip {out_name}")
            continue

        style_path = (PROJECT_ROOT / case["style_image"]).resolve()
        style_feats = engine.get_style_features(style_path)

        result = engine.paper_dual_path_generate(
            case["prompt"],
            style_feats,
            pfb_feature_indices=[2],
            rank=None,
            alpha=1.0,
            style_strength=1.0,
            style_decay=1.0,
            sac_prediction_start=2,
            sac_strength=1.0,
            enable_sac=True,
        )
        save_image_tensor(result["stylized_image_01"].detach().float().cpu(), out_path)
        progress.set_postfix_str(f"Saved {out_name}")

    # Write HTML if shard 0 or master
    if args.shard_index == 0:
        for c in TEASER_CASES:
            slug = slugify(c["content_prompt"])
            c["saved_rel_path"] = f"row{c['row']}_{c['col']:02d}_{slug}.png"
        html_content = build_teaser_html(output_dir, TEASER_CASES)
        html_file = output_dir / "index.html"
        html_file.write_text(html_content, encoding="utf-8")
        print(f"Generated HTML viewer: {html_file}")


if __name__ == "__main__":
    main()
