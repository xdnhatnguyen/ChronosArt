"""scripts/make_paper_visuals.py

Generates professional publication-quality PDF figures for report/Figure/
- Figure/qualitative_novel_methods.pdf (4x5 grid for novel methods ablation)
- Figure/benchmark_10x20_grid.pdf (4x6 comparison grid between Baseline and ChronosArt)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "report" / "Figure"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Set global matplotlib parameters for CVPR / IEEE style typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "figure.titlesize": 12,
    "figure.titleweight": "bold",
})

def make_figure_1_novel_ablation():
    """Figure 1: Ablation across 4 test cases x 5 columns (Style, Baseline, Exp4, Freq-Split, ChronosArt)."""
    cases = [
        {
            "prompt": "Prompt: 'a cute cat sitting peacefully on a cushion'",
            "style_name": "Glowing Neon",
            "style_path": PROJECT_ROOT / "styles/style_10_glowing_neon.png",
            "m0": PROJECT_ROOT / "output/novel_benchmark_showcase/case1_cat_glowing_m0_baseline_seed42.png",
            "m1": PROJECT_ROOT / "output/novel_benchmark_showcase/case1_cat_glowing_m1_champion_exp4_seed42.png",
            "m3": PROJECT_ROOT / "output/novel_benchmark_showcase/case1_cat_glowing_m3_freq_split_seed42.png",
            "ours": PROJECT_ROOT / "output/grand_synergy/case1_cat_glowing_synergy_vmod_tuned_s42.png",
        },
        {
            "prompt": "Prompt: 'a kettle sitting on a wooden table'",
            "style_name": "Colorful Origami",
            "style_path": PROJECT_ROOT / "styles/style_01_colorful_origami.png",
            "m0": PROJECT_ROOT / "output/novel_benchmark_showcase/case2_kettle_origami_m0_baseline_seed42.png",
            "m1": PROJECT_ROOT / "output/novel_benchmark_showcase/case2_kettle_origami_m1_champion_exp4_seed42.png",
            "m3": PROJECT_ROOT / "output/novel_benchmark_showcase/case2_kettle_origami_m3_freq_split_seed42.png",
            "ours": PROJECT_ROOT / "output/grand_synergy/case2_kettle_origami_synergy_vmod_tuned_s42.png",
        },
        {
            "prompt": "Prompt: 'a wild bear standing in deep forest'",
            "style_name": "Kid Crayon",
            "style_path": PROJECT_ROOT / "styles/style_04_kid_crayon_drawing.png",
            "m0": PROJECT_ROOT / "output/novel_benchmark_showcase/case3_bear_crayon_m0_baseline_seed42.png",
            "m1": PROJECT_ROOT / "output/novel_benchmark_showcase/case3_bear_crayon_m1_champion_exp4_seed42.png",
            "m3": PROJECT_ROOT / "output/novel_benchmark_showcase/case3_bear_crayon_m3_freq_split_seed42.png",
            "ours": PROJECT_ROOT / "output/grand_synergy/case3_bear_crayon_synergy_vmod_tuned_s42.png",
        },
        {
            "prompt": "Prompt: 'a wise owl perched on a branch'",
            "style_name": "Pencil Sketch",
            "style_path": PROJECT_ROOT / "styles/08_pencil_sketch.png",
            "m0": PROJECT_ROOT / "output/novel_benchmark_showcase/case4_owl_sketch_m0_baseline_seed42.png",
            "m1": PROJECT_ROOT / "output/novel_benchmark_showcase/case4_owl_sketch_m1_champion_exp4_seed42.png",
            "m3": PROJECT_ROOT / "output/novel_benchmark_showcase/case4_owl_sketch_m3_freq_split_seed42.png",
            "ours": PROJECT_ROOT / "output/grand_synergy/case4_owl_sketch_synergy_vmod_tuned_s42.png",
        },
    ]

    col_headers = [
        "Style Reference\n$I^{\\mathrm{sty}}$",
        "Baseline (CVPR 2026)\nPaper SVD ($r=8$)",
        "Exp 4 (Decay)\nRank 4 ($\\alpha=0.50$)",
        "Freq-Split PFB\nLaplacian Geometry",
        "ChronosArt\n(Proposed)",
    ]

    n_rows = len(cases)
    n_cols = len(col_headers)

    # Increased figure height and row spacing (hspace=0.38) to guarantee zero overlapping text
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 9.6), gridspec_kw={"wspace": 0.04, "hspace": 0.38})

    for row_idx, case in enumerate(cases):
        img_paths = [case["style_path"], case["m0"], case["m1"], case["m3"], case["ours"]]

        for col_idx, (ax, p) in enumerate(zip(axes[row_idx], img_paths)):
            img = Image.open(p).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Top headers on first row
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], pad=6, fontsize=9.0, fontweight="bold")

            # Highlight our ChronosArt method column with distinct green border
            if col_idx == 4:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#28a745")
                    spine.set_linewidth(2.2)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#cccccc")
                    spine.set_linewidth(0.8)

            # Left prompt annotation
            if col_idx == 0:
                ax.set_ylabel(case["style_name"], fontsize=9.0, fontweight="bold", labelpad=6)

        # Row-level prompt subtitle: perfectly centered halfway in the whitespace band
        # hspace=0.38, so -0.19 is exactly in the vertical center between rows
        axes[row_idx, 2].text(
            0.5, -0.19, case["prompt"],
            transform=axes[row_idx, 2].transAxes,
            ha="center", va="center", fontsize=8.5, style="italic", color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#f8f9fa",
                edgecolor="#d1d5db",
                linewidth=0.7,
            )
        )

    out_pdf = FIGURE_DIR / "qualitative_novel_methods.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated: {out_pdf}")


def make_figure_2_benchmark_10x20_grid():
    """Figure 2: Showcase grid comparing Baseline vs ChronosArt on top-performing pairs."""
    showcase_pairs = [
        # (prompt_id, prompt_label, style_id, style_name, style_rel_path)
        ("parti_006", "A Shiba Inu", "style_19", "Melting Golden 3D", "styles/style_19_melting_golden_3d_rendering.png"),
        ("parti_009", "A Bowl of Pho", "style_10", "Glowing Neon", "styles/style_10_glowing_neon.png"),
        ("parti_009", "A Bowl of Pho", "style_01", "Colorful Origami", "styles/style_01_colorful_origami.png"),
        ("parti_005", "Hot Air Balloon", "style_19", "Melting Golden 3D", "styles/style_19_melting_golden_3d_rendering.png"),
        ("parti_017", "A Modern Chair", "style_14", "Neon Splash Comic", "styles/style_14_neon_splash_comic_art.png"),
        ("parti_006", "A Shiba Inu", "style_11", "Line-Art Sketch", "styles/style_11_intricate_line_art_illustration.png"),
        ("parti_020", "A Robot", "style_16", "Retro Sci-Fi", "styles/style_16_retro_sci_fi_graphic.png"),
        ("parti_024", "A Boat on Water", "style_07", "Watercolor", "styles/style_07_watercolor_painting.png"),
    ]

    base_dir = PROJECT_ROOT / "output/benchmark_10x20/baseline"
    chrono_dir = PROJECT_ROOT / "output/benchmark_10x20/chronosart"
    target_dir = chrono_dir if (chrono_dir.exists() and len(list(chrono_dir.glob("*.png"))) >= 50) else (PROJECT_ROOT / "output/benchmark_10x20/proposed_vmod")

    # 4 rows x 6 columns (2 test cases per row: [Style, Baseline, ChronosArt] x 2)
    fig, axes = plt.subplots(4, 6, figsize=(11.5, 8.6), gridspec_kw={"wspace": 0.05, "hspace": 0.28})

    for i in range(4):
        pair_left = showcase_pairs[2 * i]
        pair_right = showcase_pairs[2 * i + 1]

        # Left pair: cols 0, 1, 2
        p_id_l, p_lbl_l, s_id_l, s_name_l, s_rel_l = pair_left
        s_img_l = Image.open(PROJECT_ROOT / s_rel_l)
        b_img_l = Image.open(base_dir / f"{p_id_l}_{s_id_l}.png")
        v_img_l = Image.open(target_dir / f"{p_id_l}_{s_id_l}.png")

        # Right pair: cols 3, 4, 5
        p_id_r, p_lbl_r, s_id_r, s_name_r, s_rel_r = pair_right
        s_img_r = Image.open(PROJECT_ROOT / s_rel_r)
        b_img_r = Image.open(base_dir / f"{p_id_r}_{s_id_r}.png")
        v_img_r = Image.open(target_dir / f"{p_id_r}_{s_id_r}.png")

        items = [
            (s_img_l, f"{s_name_l}", "#444444"),
            (b_img_l, "Baseline SVD", "#888888"),
            (v_img_l, "ChronosArt (Ours)", "#28a745"),
            (s_img_r, f"{s_name_r}", "#444444"),
            (b_img_r, "Baseline SVD", "#888888"),
            (v_img_r, "ChronosArt (Ours)", "#28a745"),
        ]

        for col_idx, (img, label, border_c) in enumerate(items):
            ax = axes[i, col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                hdr = "Style Ref" if col_idx in (0, 3) else ("Baseline" if col_idx in (1, 4) else "ChronosArt (Ours)")
                ax.set_title(hdr, pad=6, fontsize=9.5, fontweight="bold", color="#111111" if col_idx not in (2, 5) else "#1b5e20")

            for spine in ax.spines.values():
                spine.set_edgecolor(border_c)
                spine.set_linewidth(1.8 if border_c == "#28a745" else 0.8)

        # Labels for rows
        axes[i, 0].set_ylabel(p_lbl_l, fontsize=9, fontweight="bold", labelpad=5)
        axes[i, 3].set_ylabel(p_lbl_r, fontsize=9, fontweight="bold", labelpad=5)

    out_pdf = FIGURE_DIR / "benchmark_10x20_grid.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated: {out_pdf}")


if __name__ == "__main__":
    make_figure_1_novel_ablation()
    make_figure_2_benchmark_10x20_grid()
