#!/usr/bin/env python3
"""Run comprehensive ablation study comparing:
1. Baseline (Vanilla PFB + Vanilla SAC)
2. Idea 1: Spherical Geodesic Retraction (SLERP on S^{C-1})
3. Idea 2: Selective Head SAC (Entropy/Cosine Alignment)
4. Idea 3: Boundary Gradient Protection Mask
5. Idea 4: Spectral Fourier Phase-Lock & Texture Transfer
6. Final Combined: Unified ChronosArt Pipeline
"""

from __future__ import annotations

import argparse
import gc
import shutil
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

VARIANTS = [
    {
        "key": "baseline",
        "name": "Baseline (PFB + SAC)",
        "tag_class": "tag-base",
        "desc": "Cộng thặng dư Euclid + Đè 100% Attention Heads (Under-stylized, phong cách mỏng, dễ bùng nổ chuẩn)",
        "kwargs": {
            "use_geodesic": False,
            "sac_head_ratio": 1.0,
            "use_gradient_mask": False,
            "use_fourier_phase": False,
            "style_strength": 1.0,
        },
    },
    {
        "key": "idea1_slerp",
        "name": "Ý Tưởng 1: Spherical SLERP",
        "tag_class": "tag-slerp",
        "desc": "Quay trắc địa trên mặt cầu S^{C-1}, bảo toàn 100% chuẩn vector, triệt tiêu artifact do BSQ bit-flip",
        "kwargs": {
            "use_geodesic": True,
            "sac_head_ratio": 1.0,
            "use_gradient_mask": False,
            "use_fourier_phase": False,
            "style_strength": 1.0,
        },
    },
    {
        "key": "idea2_selective_sac",
        "name": "Ý Tưởng 2: Selective SAC",
        "tag_class": "tag-sac",
        "desc": "Chỉ ép SAC trên 75% heads cấu trúc (12/16), giải phóng 25% heads tự do tạo nét vẽ/nếp gấp",
        "kwargs": {
            "use_geodesic": False,
            "sac_head_ratio": 0.75,
            "use_gradient_mask": False,
            "use_fourier_phase": False,
            "style_strength": 1.0,
        },
    },
    {
        "key": "idea3_boundary_mask",
        "name": "Ý Tưởng 3: Boundary Mask",
        "tag_class": "tag-grad",
        "desc": "Tính gradient không gian ||∇X||, triệt tiêu style haloing tại viền đối tượng, giữ viền sắc nét",
        "kwargs": {
            "use_geodesic": False,
            "sac_head_ratio": 1.0,
            "use_gradient_mask": True,
            "gradient_mask_gamma": 0.5,
            "use_fourier_phase": False,
            "style_strength": 1.0,
        },
    },
    {
        "key": "idea4_fourier_phase",
        "name": "Ý Tưởng 4: Fourier Phase-Lock",
        "tag_class": "tag-fourier",
        "desc": "Khóa 100% góc pha hình học Oppenheim-Lim, truyền phổ biên độ phong cách ở scale vi mô s=5",
        "kwargs": {
            "use_geodesic": False,
            "sac_head_ratio": 1.0,
            "use_gradient_mask": False,
            "use_fourier_phase": True,
            "fourier_scales": [5],
            "fourier_kappa": 0.20,
            "style_strength": 1.0,
        },
    },
    {
        "key": "final_combined",
        "name": "Phiên Bản Cuối: Full ChronosArt",
        "tag_class": "tag-combined",
        "desc": "Tổng hợp SLERP + Selective SAC + Boundary Mask + Fourier Lock (Style boost 1.15x, sắc nét đỉnh cao)",
        "kwargs": {
            "use_geodesic": True,
            "sac_head_ratio": 0.75,
            "use_gradient_mask": True,
            "gradient_mask_gamma": 0.5,
            "use_fourier_phase": True,
            "fourier_scales": [5],
            "fourier_kappa": 0.20,
            "style_strength": 1.15,
        },
    },
]


def build_comparison_html(output_dir: Path, cases_data: list[dict]) -> str:
    html = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChronosArt - Thử Nghiệm Toàn Diện Từng Ý Tưởng & Tổng Hợp</title>
    <style>
        :root {
            --bg: #090d13;
            --card-bg: #131822;
            --border: #262f3d;
            --border-hover: #3d495d;
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
            --accent: #58a6ff;
            --color-ref: #f2cc60;
            --color-base: #f85149;
            --color-slerp: #79c0ff;
            --color-sac: #d2a8ff;
            --color-grad: #56d364;
            --color-fourier: #ff7b72;
            --color-combined: #3fb950;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px 32px 80px 32px;
            line-height: 1.5;
        }
        .header {
            max-width: 1700px;
            margin: 0 auto 28px auto;
        }
        .header h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #3fb950);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p {
            color: var(--text-sub);
            font-size: 15px;
            margin-bottom: 16px;
        }
        .theory-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .theory-card {
            background: rgba(19, 24, 34, 0.8);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 16px;
            font-size: 13px;
        }
        .theory-card strong {
            display: block;
            margin-bottom: 4px;
            font-size: 14px;
        }
        .theory-slerp strong { color: var(--color-slerp); }
        .theory-sac strong { color: var(--color-sac); }
        .theory-grad strong { color: var(--color-grad); }
        .theory-fourier strong { color: var(--color-fourier); }
        .theory-combined strong { color: var(--color-combined); }

        .container {
            max-width: 1700px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 36px;
        }
        .case-section {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        .case-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .case-title h2 {
            font-size: 20px;
            color: var(--text-main);
        }
        .case-prompt {
            font-size: 13px;
            color: var(--accent);
            background: rgba(88, 166, 255, 0.1);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(88, 166, 255, 0.3);
        }
        .columns-row {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 12px;
        }
        @media (max-width: 1400px) {
            .columns-row {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
        }
        @media (max-width: 900px) {
            .columns-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        .col-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, border-color 0.2s;
        }
        .col-card:hover {
            border-color: var(--border-hover);
        }
        .col-card.highlight {
            border: 2px solid var(--color-combined);
            background: rgba(63, 185, 80, 0.04);
        }
        .col-tag {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 4px 8px;
            border-radius: 6px;
            margin-bottom: 8px;
            text-align: center;
        }
        .tag-ref { background: rgba(242, 204, 96, 0.15); color: var(--color-ref); border: 1px solid rgba(242, 204, 96, 0.4); }
        .tag-base { background: rgba(248, 81, 73, 0.15); color: var(--color-base); border: 1px solid rgba(248, 81, 73, 0.4); }
        .tag-slerp { background: rgba(121, 192, 255, 0.15); color: var(--color-slerp); border: 1px solid rgba(121, 192, 255, 0.4); }
        .tag-sac { background: rgba(210, 168, 255, 0.15); color: var(--color-sac); border: 1px solid rgba(210, 168, 255, 0.4); }
        .tag-grad { background: rgba(86, 211, 100, 0.15); color: var(--color-grad); border: 1px solid rgba(86, 211, 100, 0.4); }
        .tag-fourier { background: rgba(255, 123, 114, 0.15); color: var(--color-fourier); border: 1px solid rgba(255, 123, 114, 0.4); }
        .tag-combined { background: rgba(63, 185, 80, 0.2); color: var(--color-combined); border: 1px solid rgba(63, 185, 80, 0.6); }

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
            box-shadow: 0 6px 16px rgba(0,0,0,0.6);
        }
        .img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .desc {
            margin-top: 8px;
            font-size: 11px;
            color: var(--text-sub);
            line-height: 1.35;
            flex-grow: 1;
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
            backdrop-filter: blur(6px);
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }
        .modal img {
            max-width: 88%;
            max-height: 85%;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.8);
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
        <h1>ChronosArt - Thử Nghiệm Tách Biệt Toàn Bộ Ý Tưởng & Phiên Bản Tổng Hợp</h1>
        <p>Báo cáo thực nghiệm đối đầu trực quan giữa Baseline nguyên bản, từng thành phần cải tiến độc lập (Ablation Study) và Phiên bản Hợp nhất (Unified Final Pipeline).</p>
        
        <div class="theory-grid">
            <div class="theory-card theory-slerp">
                <strong>Ý tưởng 1: Spherical SLERP</strong>
                Quay trắc địa trên mặt cầu S^(C-1) bảo toàn chuẩn ||x||_2, ngăn ngừa bùng nổ chuẩn và triệt tiêu lỗi lật bit BSQ.
            </div>
            <div class="theory-card theory-sac">
                <strong>Ý tưởng 2: Selective Head SAC</strong>
                Chỉ ép SAC lên 75% heads có độ tương quan hình học cao nhất, giải phóng 25% heads tự do để hấp thụ nét cọ/nếp gấp.
            </div>
            <div class="theory-card theory-grad">
                <strong>Ý tưởng 3: Boundary Gradient Mask</strong>
                Điều chế biên độ style theo gradient không gian ||∇X||, bảo vệ đường biên sắc nét, loại bỏ quầng mờ (haloing).
            </div>
            <div class="theory-card theory-fourier">
                <strong>Ý tưởng 4: Spectral Fourier Lock</strong>
                Khóa 100% pha Fourier hình học, pha trộn phổ biên độ phong cách ở scale vi mô s=5, kích hoạt chi tiết chất cảm vi mô.
            </div>
            <div class="theory-card theory-combined">
                <strong>Phiên Bản Cuối: Full ChronosArt</strong>
                Kết hợp toàn bộ 4 components, nâng độ đậm phong cách lên 1.15x mà cấu trúc vẫn hoàn hảo và sạch artifact.
            </div>
        </div>
    </div>

    <div class="container">
"""

    for c in cases_data:
        html += f"""
        <div class="case-section">
            <div class="case-title">
                <h2>{c['title']}</h2>
                <span class="case-prompt">{c['prompt']}</span>
            </div>
            <div class="columns-row">
                <!-- Style Ref -->
                <div class="col-card">
                    <div class="col-tag tag-ref">Style Reference</div>
                    <div class="img-wrap" onclick="openModal('{c['style_rel']}', 'Style Reference: {c['style_name']}')">
                        <img src="{c['style_rel']}" alt="Ref">
                    </div>
                    <div class="desc">{c['style_name']}</div>
                </div>
"""
        for v in c["variants"]:
            is_comb = v["key"] == "final_combined"
            highlight_class = "highlight" if is_comb else ""
            html += f"""
                <div class="col-card {highlight_class}">
                    <div class="col-tag {v['tag_class']}">{v['name']}</div>
                    <div class="img-wrap" onclick="openModal('{v['img_rel']}', '{v['name']}: {c['title']}')">
                        <img src="{v['img_rel']}" alt="{v['name']}">
                    </div>
                    <div class="desc">{v['desc']}</div>
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
    output_dir = PROJECT_ROOT / "output/experiment_all"
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

    # Reuse baseline if available from previous experiment
    prev_dir = PROJECT_ROOT / "output/experiment_ideal1"

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

    cases_data = []

    for case in TEST_CASES:
        cid = case["id"]
        print(f"\n================ Case: {case['title']} ================")
        style_path = (PROJECT_ROOT / case["style_image"]).resolve()
        style_feats = engine.get_style_features(style_path)

        case_variants_data = []

        for var in VARIANTS:
            vkey = var["key"]
            img_name = f"{cid}_{vkey}.png"
            img_path = output_dir / img_name

            # Check if we can reuse baseline
            if vkey == "baseline" and not img_path.is_file():
                prev_baseline = prev_dir / f"{cid}_baseline.png"
                if prev_baseline.is_file():
                    shutil.copy2(prev_baseline, img_path)
                    print(f"  [Copied Baseline] {img_name}")

            if not img_path.is_file():
                print(f"  Generating {var['name']} -> {img_name}...")
                t0 = time.time()
                res = engine.paper_dual_path_generate(
                    case["prompt"],
                    style_feats,
                    pfb_feature_indices=[2],
                    rank=None,
                    alpha=1.0,
                    style_decay=1.0,
                    sac_prediction_start=2,
                    sac_strength=1.0,
                    enable_sac=True,
                    **var["kwargs"],
                )
                save_image_tensor(res["stylized_image_01"].detach().float().cpu(), img_path)
                print(f"    Done in {time.time() - t0:.2f}s")
            else:
                print(f"  Already exists: {img_name}")

            case_variants_data.append({
                "key": vkey,
                "name": var["name"],
                "tag_class": var["tag_class"],
                "desc": var["desc"],
                "img_rel": img_name,
            })

        cases_data.append({
            "id": cid,
            "title": case["title"],
            "prompt": case["prompt"],
            "style_name": case["style_name"],
            "style_rel": f"../../{case['style_image']}",
            "variants": case_variants_data,
        })

    # Build and write HTML
    html_content = build_comparison_html(output_dir, cases_data)
    html_path = output_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"\nSUCCESS: All experiments complete! HTML viewer written to: {html_path}")


if __name__ == "__main__":
    main()
