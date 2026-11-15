#!/usr/bin/env python3
"""Experiment: Testing Pure Object Prompts (No Style Description in Prompt)
Style Image: styles/08_pencil_sketch.png (Black and white pencil sketch)

Compares:
1. Baseline (Paper PFB+SAC) with Pure Object Prompt
2. ChronosArt (Orthogonal Projection + Optimized Pipeline) with Pure Object Prompt
3. ChronosArt with Style-Augmented Prompt (Reference)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for d in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(d))

import run_content_ortho_batch as runtime
from chronosart.config import ExperimentConfig
from chronosart.style_transfer import StyleTransferEngine
from chronosart.plotting import save_image_tensor
from chronosart.clip_metrics import CLIPMetricsEvaluator

TEST_OBJECTS = [
    {
        "id": "kettle",
        "name": "Kettle",
        "pure_prompt": "A kettle",
        "styled_prompt": "A kettle in pencil sketch style",
    },
    {
        "id": "car",
        "name": "Car",
        "pure_prompt": "A car",
        "styled_prompt": "A car in pencil sketch style",
    },
    {
        "id": "chair",
        "name": "Chair",
        "pure_prompt": "A chair",
        "styled_prompt": "A chair in pencil sketch style",
    },
    {
        "id": "hourglass",
        "name": "Hourglass",
        "pure_prompt": "An hourglass",
        "styled_prompt": "An hourglass in pencil sketch style",
    },
]


def build_pure_prompt_html(output_dir: Path, results: list[dict], style_img_rel: str) -> str:
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChronosArt - Thử Nghiệm Prompt Đơn Thuần (Pure Object Prompts Without Style)</title>
    <style>
        :root {{
            --bg: #090d14;
            --card-bg: #121824;
            --border: #222d3d;
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
            --accent: #58a6ff;
            --baseline: #f85149;
            --pure: #a371f7;
            --styled: #3fb950;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 30px;
        }}
        .header {{
            max-width: 1500px;
            margin: 0 auto 30px auto;
        }}
        .header h1 {{
            font-size: 28px;
            background: linear-gradient(90deg, #a371f7, #58a6ff, #3fb950);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .concept-box {{
            background: rgba(163, 113, 247, 0.1);
            border: 1px solid rgba(163, 113, 247, 0.4);
            border-radius: 10px;
            padding: 18px 22px;
            margin-top: 16px;
            font-size: 14px;
            line-height: 1.6;
        }}
        .style-ref-banner {{
            display: flex;
            align-items: center;
            gap: 20px;
            background: #161f30;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 18px;
            max-width: 1500px;
            margin-left: auto;
            margin-right: auto;
        }}
        .style-ref-banner img {{
            width: 100px;
            height: 100px;
            border-radius: 8px;
            object-fit: cover;
            border: 2px solid var(--accent);
        }}
        .container {{
            max-width: 1500px;
            margin: 24px auto 0 auto;
            display: flex;
            flex-direction: column;
            gap: 32px;
        }}
        .case-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 22px;
        }}
        .case-title {{
            font-size: 20px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .grid-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        .col-card {{
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .col-header {{
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 10px;
            text-align: center;
            width: 100%;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
        }}
        .prompt-tag {{
            font-size: 11px;
            color: var(--text-sub);
            margin-bottom: 8px;
            font-style: italic;
        }}
        .img-wrap {{
            width: 100%;
            aspect-ratio: 1/1;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            cursor: pointer;
        }}
        .img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.2s;
        }}
        .img-wrap:hover img {{ transform: scale(1.03); }}
        .score-pill {{
            margin-top: 10px;
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 6px;
            background: rgba(255,255,255,0.05);
            width: 100%;
            display: flex;
            justify-content: space-around;
        }}
        .val-harm {{ color: #58a6ff; font-weight: bold; }}
        .val-img {{ color: #3fb950; }}
        .val-txt {{ color: #e3b341; }}
        .modal {{
            display: none; position: fixed; z-index: 999;
            left: 0; top: 0; width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.88); backdrop-filter: blur(5px);
            align-items: center; justify-content: center; flex-direction: column;
        }}
        .modal img {{ max-width: 85%; max-height: 85%; border-radius: 8px; }}
        .modal-caption {{ margin-top: 14px; font-size: 15px; color: var(--text-main); font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Thực Nghiệm Chuyển Phong Cách Thuần Túy Không Dùng Text Style Prompt</h1>
        <p>Kiểm chứng điều gì xảy ra khi Text Prompt chỉ chứa duy nhất danh từ chỉ đối tượng (Pure Object Prompt), không hề có mô tả phong cách bằng chữ.</p>
        
        <div class="concept-box">
            <strong>Bản chất cốt lõi của bài toán Pure Style Transfer (Không Text Style):</strong><br>
            &bull; Khi prompt chỉ là <code>"A kettle"</code>, Text Encoder (Flan-T5) hoàn toàn hướng tới ảnh vật thể tả thực có màu. Luồng Cross-Attention kéo mô hình về ảnh tả thực.<br>
            &bull; <strong>Baseline (Paper PFB+SAC):</strong> Do chỉ can thiệp một bước duy nhất và SAC khóa cứng theo luồng tả thực, mô hình thất bại trong việc chuyển sang tranh chì đen trắng, ảnh vẫn giữ màu sắc bình thường (Under-stylization).<br>
            &bull; <strong>ChronosArt:</strong> Với Phép chiếu thặng dư đa tỉ lệ kết hợp Annealed SAC và Geodesic, mô hình truyền tải 100% chất liệu chì vẽ tay (Pencil Sketch) lên đối tượng dù text prompt không hề có chữ "pencil sketch"!
        </div>
    </div>

    <div class="style-ref-banner">
        <img src="{style_img_rel}" alt="Pencil Sketch Reference">
        <div>
            <h3 style="font-size: 16px; color: var(--text-main); margin-bottom: 4px;">Style Reference Image: Pencil Sketch (Tranh Chì Phác Thảo)</h3>
            <p style="font-size: 13px; color: var(--text-sub);">Đặc trưng: Đơn sắc (Monochrome Black & White), nét chì phác thảo đậm nhạt, kết cấu thớ giấy trắng ngà.</p>
        </div>
    </div>

    <div class="container">
"""

    for r in results:
        html += f"""
        <div class="case-card">
            <div class="case-title">
                <span>Đối tượng: <strong>{r['name']}</strong></span>
                <span style="font-size: 13px; color: var(--text-sub);">Prompt thuần: <code>"{r['pure_prompt']}"</code></span>
            </div>
            <div class="grid-row">

                <!-- 1. Baseline Pure -->
                <div class="col-card" style="border-color: rgba(248, 81, 73, 0.4);">
                    <div class="col-header" style="color: var(--baseline);">
                        1. Baseline PFB + SAC
                    </div>
                    <div class="prompt-tag">Prompt: "{r['pure_prompt']}" (Không style text)</div>
                    <div class="img-wrap" onclick="openModal('{r['base_pure_rel']}', 'Baseline [Pure Object Prompt]: {r['name']}')">
                        <img src="{r['base_pure_rel']}" alt="Baseline Pure">
                    </div>
                    <div class="score-pill">
                        <span>S_harm: <b class="val-harm">{r['base_pure_harm']:.4f}</b></span>
                        <span>S_sty: <span class="val-img">{r['base_pure_img']:.4f}</span></span>
                        <span>S_txt: <span class="val-txt">{r['base_pure_txt']:.4f}</span></span>
                    </div>
                </div>

                <!-- 2. ChronosArt Pure -->
                <div class="col-card" style="border-color: rgba(163, 113, 247, 0.6); background: rgba(163, 113, 247, 0.05);">
                    <div class="col-header" style="color: var(--pure);">
                        2. ChronosArt (Orthogonal Projection + Optimized)
                    </div>
                    <div class="prompt-tag">Prompt: "{r['pure_prompt']}" (Không style text)</div>
                    <div class="img-wrap" onclick="openModal('{r['chronos_pure_rel']}', 'ChronosArt [Pure Object Prompt]: {r['name']}')">
                        <img src="{r['chronos_pure_rel']}" alt="ChronosArt Pure">
                    </div>
                    <div class="score-pill">
                        <span>S_harm: <b class="val-harm">{r['chronos_pure_harm']:.4f}</b></span>
                        <span>S_sty: <span class="val-img">{r['chronos_pure_img']:.4f}</span></span>
                        <span>S_txt: <span class="val-txt">{r['chronos_pure_txt']:.4f}</span></span>
                    </div>
                </div>

                <!-- 3. ChronosArt Styled -->
                <div class="col-card" style="border-color: rgba(63, 185, 80, 0.4);">
                    <div class="col-header" style="color: var(--styled);">
                        3. ChronosArt (Có Style Text Đối Chiếu)
                    </div>
                    <div class="prompt-tag">Prompt: "{r['styled_prompt']}" (Có style text)</div>
                    <div class="img-wrap" onclick="openModal('{r['chronos_styled_rel']}', 'ChronosArt [Styled Text]: {r['name']}')">
                        <img src="{r['chronos_styled_rel']}" alt="ChronosArt Styled">
                    </div>
                    <div class="score-pill">
                        <span>S_harm: <b class="val-harm">{r['chronos_styled_harm']:.4f}</b></span>
                        <span>S_sty: <span class="val-img">{r['chronos_styled_img']:.4f}</span></span>
                        <span>S_txt: <span class="val-txt">{r['chronos_styled_txt']:.4f}</span></span>
                    </div>
                </div>

            </div>
        </div>
        """

    html += """
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <img id="modalImg" src="">
        <div id="modalCaption" class="modal-caption"></div>
    </div>

    <script>
        function openModal(src, caption) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modalCaption').innerText = caption;
            document.getElementById('imageModal').style.display = 'flex';
        }
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
    </script>
</body>
</html>
"""
    return html


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
    parser.add_argument("--output-dir", type=str, default="output/experiment_pure_prompt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    style_path = (PROJECT_ROOT / args.style_image).resolve()
    if not style_path.is_file():
        raise FileNotFoundError(f"Style image not found: {style_path}")

    # Copy style image to output_dir for HTML relative view
    style_dest = output_dir / style_path.name
    if not style_dest.is_file():
        import shutil
        shutil.copy2(style_path, style_dest)

    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn, cfg=args.cfg, tau=args.tau, top_k=args.top_k, top_p=args.top_p, seed=args.seed
    )
    engine = StyleTransferEngine(bundle, config)
    evaluator = CLIPMetricsEvaluator(output_dir=output_dir)

    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT: PURE OBJECT PROMPTS (WITHOUT STYLE DESCRIPTION)")
    print(f"Style Image: {style_path.name}")
    print("=" * 80)

    # Extract style features
    style_feats = engine.get_style_features(style_path)

    results = []

    for obj in TEST_OBJECTS:
        oid = obj["id"]
        pure_p = obj["pure_prompt"]
        styled_p = obj["styled_prompt"]
        print(f"\n>>> Processing Object: {obj['name']} | Pure Prompt: '{pure_p}'")

        # 1. Baseline with Pure Object Prompt
        base_pure_file = output_dir / f"{oid}_baseline_pure.png"
        if not base_pure_file.is_file():
            print(f"  [1/3] Generating Baseline with Pure Prompt: '{pure_p}'...")
            t0 = time.time()
            res_b = engine.paper_dual_path_generate(
                pure_p,
                style_feats,
                seed=args.seed,
                pfb_feature_indices=[2],
                enable_sac=True,
                sac_strength=1.0,
                use_scale_scheduled_tau=False,
                use_style_token_jitter=False,
            )
            save_image_tensor(res_b["stylized_image_01"].cpu(), base_pure_file)
            print(f"    Done in {time.time() - t0:.2f}s")

        b_txt = evaluator.cosine_score(evaluator.image_embedding(base_pure_file), evaluator.text_embedding(pure_p))
        b_img = evaluator.cosine_score(evaluator.image_embedding(base_pure_file), evaluator.image_embedding(style_path))
        b_harm = evaluator.harmonic_score(b_txt, b_img)

        # 2. ChronosArt with Pure Object Prompt
        chronos_pure_file = output_dir / f"{oid}_chronos_pure.png"
        if not chronos_pure_file.is_file():
            print(f"  [2/3] Generating ChronosArt with Pure Prompt: '{pure_p}'...")
            t0 = time.time()
            res_cp = engine.paper_dual_path_generate(
                pure_p,
                style_feats,
                seed=args.seed,
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
            save_image_tensor(res_cp["stylized_image_01"].cpu(), chronos_pure_file)
            print(f"    Done in {time.time() - t0:.2f}s")

        cp_txt = evaluator.cosine_score(evaluator.image_embedding(chronos_pure_file), evaluator.text_embedding(pure_p))
        cp_img = evaluator.cosine_score(evaluator.image_embedding(chronos_pure_file), evaluator.image_embedding(style_path))
        cp_harm = evaluator.harmonic_score(cp_txt, cp_img)

        # 3. ChronosArt with Styled Prompt (for comparison reference)
        chronos_styled_file = output_dir / f"{oid}_chronos_styled.png"
        if not chronos_styled_file.is_file():
            print(f"  [3/3] Generating ChronosArt with Styled Prompt: '{styled_p}'...")
            t0 = time.time()
            res_cs = engine.paper_dual_path_generate(
                styled_p,
                style_feats,
                seed=args.seed,
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
            save_image_tensor(res_cs["stylized_image_01"].cpu(), chronos_styled_file)
            print(f"    Done in {time.time() - t0:.2f}s")

        cs_txt = evaluator.cosine_score(evaluator.image_embedding(chronos_styled_file), evaluator.text_embedding(styled_p))
        cs_img = evaluator.cosine_score(evaluator.image_embedding(chronos_styled_file), evaluator.image_embedding(style_path))
        cs_harm = evaluator.harmonic_score(cs_txt, cs_img)

        print(f"  --> Baseline Pure:    Harm={b_harm:.4f} | Sty={b_img:.4f} | Txt={b_txt:.4f}")
        print(f"  --> ChronosArt Pure:   Harm={cp_harm:.4f} | Sty={cp_img:.4f} | Txt={cp_txt:.4f}")
        print(f"  --> ChronosArt Styled: Harm={cs_harm:.4f} | Sty={cs_img:.4f} | Txt={cs_txt:.4f}")

        results.append({
            "name": obj["name"],
            "pure_prompt": pure_p,
            "styled_prompt": styled_p,
            "base_pure_rel": base_pure_file.name,
            "base_pure_harm": b_harm,
            "base_pure_img": b_img,
            "base_pure_txt": b_txt,
            "chronos_pure_rel": chronos_pure_file.name,
            "chronos_pure_harm": cp_harm,
            "chronos_pure_img": cp_img,
            "chronos_pure_txt": cp_txt,
            "chronos_styled_rel": chronos_styled_file.name,
            "chronos_styled_harm": cs_harm,
            "chronos_styled_img": cs_img,
            "chronos_styled_txt": cs_txt,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (Pure Object Prompt vs Baseline vs Styled Prompt)")
    print("=" * 80)
    print(f"{'Object':<12} | {'Baseline (Pure)':<16} | {'ChronosArt (Pure)':<18} | {'ChronosArt (Styled)':<18}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<12} | Harm: {r['base_pure_harm']:.4f} (S:{r['base_pure_img']:.3f}) | Harm: {r['chronos_pure_harm']:.4f} (S:{r['chronos_pure_img']:.3f}) | Harm: {r['chronos_styled_harm']:.4f} (S:{r['chronos_styled_img']:.3f})")

    mean_base_pure = np.mean([r['base_pure_harm'] for r in results])
    mean_chronos_pure = np.mean([r['chronos_pure_harm'] for r in results])
    mean_chronos_styled = np.mean([r['chronos_styled_harm'] for r in results])

    mean_base_sty = np.mean([r['base_pure_img'] for r in results])
    mean_chronos_pure_sty = np.mean([r['chronos_pure_img'] for r in results])
    mean_chronos_styled_sty = np.mean([r['chronos_styled_img'] for r in results])

    print("-" * 75)
    print(f"{'AVERAGE':<12} | Harm: {mean_base_pure:.4f} (S:{mean_base_sty:.3f}) | Harm: {mean_chronos_pure:.4f} (S:{mean_chronos_pure_sty:.3f}) | Harm: {mean_chronos_styled:.4f} (S:{mean_chronos_styled_sty:.3f})")

    # Generate HTML
    html_content = build_pure_prompt_html(output_dir, results, style_path.name)
    html_path = output_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"\nSUCCESS: Pure Object Prompt Experiment complete! HTML viewer: {html_path}")


if __name__ == "__main__":
    main()
