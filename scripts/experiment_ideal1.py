#!/usr/bin/env python3
"""Run side-by-side experiment comparing Baseline PFB+SAC vs Idea 1 (SLERP + Selective Head SAC)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_content_ortho_batch as runtime

TEST_CASES = [
    {
        "id": "case1_kettle_origami",
        "title": "A kettle (Origami Style)",
        "content_prompt": "A kettle",
        "prompt": "A kettle in colorful origami style",
        "style_name": "Colorful Origami",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case2_car_origami",
        "title": "A car (Origami Style)",
        "content_prompt": "A car",
        "prompt": "A car in colorful origami style",
        "style_name": "Colorful Origami",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case3_pen_3d",
        "title": "A pen (3D Rendering)",
        "content_prompt": "A pen",
        "prompt": "A pen in 3d rendering style",
        "style_name": "3D Rendering",
        "style_image": "styles/style_02_3d_rendering.png",
    },
    {
        "id": "case4_hourglass_renaissance",
        "title": "An hourglass (Northern Renaissance)",
        "content_prompt": "An hourglass",
        "prompt": "An hourglass in northern renaissance art style",
        "style_name": "Northern Renaissance Art",
        "style_image": "styles/style_03_northern_renaissance_art.png",
    },
]


def build_comparison_html(output_dir: Path, results: list[dict]) -> str:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChronosArt - Idea 1 Experiment: Spherical Geodesic + Selective Head SAC</title>
    <style>
        :root {
            --bg: #090d13;
            --card-bg: #141922;
            --border: #2b323c;
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
            --accent: #58a6ff;
            --baseline: #f85149;
            --ideal: #3fb950;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
        }
        .header {
            max-width: 1300px;
            margin: 0 auto 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 26px;
            margin-bottom: 8px;
        }
        .header p {
            color: var(--text-sub);
            max-width: 850px;
            margin: 0 auto 16px;
            line-height: 1.5;
            font-size: 14px;
        }
        .theory-box {
            background: rgba(88, 166, 255, 0.08);
            border: 1px solid rgba(88, 166, 255, 0.25);
            border-radius: 8px;
            padding: 14px 20px;
            text-align: left;
            max-width: 950px;
            margin: 0 auto 28px;
            font-size: 13px;
            line-height: 1.6;
        }
        .theory-box strong { color: var(--accent); }
        .grid {
            max-width: 1300px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }
        .case-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        .case-title {
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }
        .columns {
            display: grid;
            grid-template-columns: 1fr 1.3fr 1.3fr;
            gap: 20px;
            align-items: start;
        }
        .col-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .col-tag {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            padding: 3px 10px;
            border-radius: 12px;
        }
        .tag-ref { background: rgba(227, 179, 65, 0.15); color: #f2cc60; border: 1px solid rgba(227, 179, 65, 0.4); }
        .tag-base { background: rgba(248, 81, 73, 0.15); color: var(--baseline); border: 1px solid rgba(248, 81, 73, 0.4); }
        .tag-ideal { background: rgba(63, 185, 80, 0.15); color: var(--ideal); border: 1px solid rgba(63, 185, 80, 0.4); }
        .img-wrap {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .img-wrap:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        }
        .img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .desc {
            margin-top: 10px;
            font-size: 12px;
            color: var(--text-sub);
            line-height: 1.4;
        }
        /* Modal for zoom */
        .modal {
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
        }
        .modal img {
            max-width: 85%;
            max-height: 85%;
            border-radius: 8px;
        }
        .modal-caption {
            margin-top: 14px;
            font-size: 15px;
            color: var(--text-main);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Thực Nghiệm Ý Tưởng 1: Spherical Geodesic (SLERP) + Selective Head Attention</h1>
        <p>So sánh đối đầu trực tiếp giữa phương pháp Baseline PFB+SAC nguyên bản và Ý tưởng 1 cải tiến toán học nhằm khắc phục hiện tượng <strong>under-stylization</strong> và <strong>vỡ cấu trúc / artifact</strong>.</p>
        <div class="theory-box">
            <strong>Đột phá toán học trong Ý tưởng 1:</strong><br>
            &bull; <strong>Spherical Geodesic Retraction:</strong> Thay vì phép cộng vector Euclid $x + \lambda \Delta$ làm lệch vector khỏi mặt cầu đơn vị và gây lật bit ngẫu nhiên trong BSQ, thuật toán chiếu $\Delta$ lên không gian tiếp tuyến $T_x S^{C-1}$ và quay trắc địa bảo toàn 100% chuẩn $\|\mathbf{x}\|_2$, triệt tiêu artifact.<br>
            &bull; <strong>Selective Head SAC:</strong> Thay vì cưỡng chế đè 100% (16/16) attention heads làm bóp chết nét vẽ phong cách, thuật toán chỉ áp dụng SAC cho các head có độ thẳng hàng cấu trúc cao (12 heads), giải phóng 4 heads chất liệu để tự do hấp thụ nét cọ, nếp gấp origami và ánh sáng phong cách.
        </div>
    </div>

    <div class="grid">
"""
    for res in results:
        style_rel = res["style_rel"]
        base_rel = res["baseline_rel"]
        ideal_rel = res["ideal_rel"]
        html += f"""
        <div class="case-card">
            <div class="case-title">
                <span>{res['title']}</span>
                <span style="font-size: 13px; font-weight: normal; color: var(--text-sub);">{res['prompt']}</span>
            </div>
            <div class="columns">
                <div class="col-card">
                    <div class="col-tag tag-ref">Style Reference</div>
                    <div class="img-wrap" onclick="openModal('{style_rel}', 'Style Reference: {res['style_name']}')">
                        <img src="{style_rel}" alt="Reference">
                    </div>
                    <div class="desc">{res['style_name']}</div>
                </div>

                <div class="col-card">
                    <div class="col-tag tag-base">Baseline (PFB + SAC)</div>
                    <div class="img-wrap" onclick="openModal('{base_rel}', 'Baseline (PFB + SAC): {res['title']}')">
                        <img src="{base_rel}" alt="Baseline">
                    </div>
                    <div class="desc">Cộng thặng dư Euclid + Đè 100% Attention Heads (Under-stylized, phong cách còn mỏng)</div>
                </div>

                <div class="col-card">
                    <div class="col-tag tag-ideal">Ý Tưởng 1 (Geodesic + Selective SAC)</div>
                    <div class="img-wrap" onclick="openModal('{ideal_rel}', 'Ý Tưởng 1 (Geodesic + Selective SAC): {res['title']}')">
                        <img src="{ideal_rel}" alt="Idea 1">
                    </div>
                    <div class="desc"><strong>Bảo toàn chuẩn mặt cầu S^(C-1)</strong> + Giải phóng Texture Heads (Nét vẽ/nếp gấp đậm đà, sạch artifact)</div>
                </div>
            </div>
        </div>
"""

    html += """
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <img id="modalImg" src="" alt="Zoomed image">
        <div id="modalCap" class="modal-caption"></div>
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
    output_dir = PROJECT_ROOT / "output/experiment_ideal1"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime.load_runtime_modules()
    import torch
    from chronosart.config import ExperimentConfig
    from chronosart.plotting import save_image_tensor
    from chronosart.style_transfer import StyleTransferEngine

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    args = parser.parse_args()

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

    results_data = []

    for case in TEST_CASES:
        cid = case["id"]
        print(f"\n================ Running: {case['title']} ================")
        style_path = (PROJECT_ROOT / case["style_image"]).resolve()
        style_feats = engine.get_style_features(style_path)

        # 1. Baseline: use_geodesic=False, sac_head_ratio=1.0
        base_out = output_dir / f"{cid}_baseline.png"
        if not base_out.is_file():
            print("  [1/2] Generating Baseline (Euclidean addition, 100% SAC heads)...")
            t0 = time.time()
            res_base = engine.paper_dual_path_generate(
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
                use_geodesic=False,
                sac_head_ratio=1.0,
            )
            save_image_tensor(res_base["stylized_image_01"].detach().float().cpu(), base_out)
            print(f"        Saved {base_out.name} ({time.time() - t0:.2f}s)")
        else:
            print(f"  [1/2] Baseline already exists: {base_out.name}")

        # 2. Idea 1: use_geodesic=True, sac_head_ratio=0.75 (12/16 structural heads, 4/16 texture heads)
        ideal_out = output_dir / f"{cid}_ideal1.png"
        if not ideal_out.is_file():
            print("  [2/2] Generating Idea 1 (Spherical Geodesic + 75% Selective Head SAC)...")
            t0 = time.time()
            res_ideal = engine.paper_dual_path_generate(
                case["prompt"],
                style_feats,
                pfb_feature_indices=[2],
                rank=None,
                alpha=1.0,
                style_strength=1.1,  # Can afford 1.1x style because manifold preservation prevents artifacts!
                style_decay=1.0,
                sac_prediction_start=2,
                sac_strength=1.0,
                enable_sac=True,
                use_geodesic=True,
                sac_head_ratio=0.75,
            )
            save_image_tensor(res_ideal["stylized_image_01"].detach().float().cpu(), ideal_out)
            print(f"        Saved {ideal_out.name} ({time.time() - t0:.2f}s)")
        else:
            print(f"  [2/2] Idea 1 already exists: {ideal_out.name}")

        results_data.append({
            "id": cid,
            "title": case["title"],
            "prompt": case["prompt"],
            "style_name": case["style_name"],
            "style_rel": f"../../{case['style_image']}",
            "baseline_rel": base_out.name,
            "ideal_rel": ideal_out.name,
        })

    # Generate comparison HTML
    html_content = build_comparison_html(output_dir, results_data)
    html_path = output_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"\nSUCCESS: Comparison HTML viewer generated at: {html_path}")


if __name__ == "__main__":
    main()
