# ChronosArt

> **Reproduction & Extension of *"A Training-Free Style-Personalization via SVD-Based Feature Decomposition"***  
> Training-Free Style Personalization on Visual Autoregressive Transformers (Infinity-2B) with Safe Stochastic Generation.

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Wildlife%20Camera%20Roll-brightgreen)](https://xdnhatnguyen.github.io/ChronosArt/)
[![Paper](https://img.shields.io/badge/Literature%20Review-PDF-red)](report/Literature%20Review:%20SVD-Based%20Feature%20Decomposition.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Overview

**ChronosArt** reproduces and extends the foundational methodology of **"A Training-Free Style-Personalization via SVD-Based Feature Decomposition"** using state-of-the-art Visual Autoregressive transformers (**Infinity-2B**).

The framework transfers high-frequency artistic brushwork, textures, and color schemes from arbitrary reference images without model fine-tuning or test-time optimization, operating in **~2.0s per image** at 512×512 resolution.

### Core Architectural Components:
1. **Visual Autoregressive Modeling (Infinity-2B):** Predicts multi-scale discrete token distributions across 10 progressive resolution scales ($1\times 1 \to 64\times 64$).
2. **Principal Feature Blending (PFB):** Formulates truncated SVD with exponential singular value decay ($\Phi_{r, \alpha}(\mathbf{F}_s) = \sum_{i=1}^r e^{-\alpha(i-1)} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$) to extract dominant style representations while suppressing content interference.
3. **Structural Attention Correction (SAC):** Enforces Query-Key synchrony from a clean content inference stream into the generation self-attention layers ($\mathbf{A}_s^{\text{gen}} \leftarrow \text{softmax}(\mathbf{Q}_s^{\text{con}} (\mathbf{K}_s^{\text{con}})^\top / \sqrt{d})$), locking anatomical structure and geometry.
4. **Safe Stochastic Generation:** Introduces scale-scheduled temperature ($\tau_{\text{coarse}} = 0.15 \to \tau_{\text{fine}} \in [0.85, 1.25]$) and fine-scale spatial token jittering to sample unique micro-brushstroke variations per seed without geometric deformation.

---

## 📂 Repository Structure

```text
.
├── docs/                       # Interactive Wildlife Camera Roll web exhibition (GitHub Pages)
│   ├── assets/                 # WebP frames, favicon, and report PDF
│   ├── css/style.css           # Camera roll whiteboard styling (Camroll inspired)
│   ├── data/animals_dual.json  # 10 species metadata across 3 stochastic seeds
│   ├── js/app.js               # Draggable whiteboard and live stochastic jump engine
│   └── index.html              # Main web project page
├── third-party/
│   └── Infinity/               # FoundationVision Infinity-2B engine
├── data/
│   └── csd100/                 # CSD100 benchmark reference image pairs
├── report/
│   └── Literature Review: SVD-Based Feature Decomposition.pdf
├── src/
│   └── chronosart/             # Core Python package
│       ├── bootstrap.py        # Infinity runtime patching & model loader
│       ├── clip_metrics.py     # CLIP text alignment & style similarity metrics
│       ├── config.py           # Experiment configuration dataclasses
│       ├── dataset.py          # Dataset loading utilities
│       ├── svd_blending.py     # SVD truncated decomposition & PFB implementation
│       ├── style_transfer.py   # Full dual-stream PFB + SAC inference engine
│       ├── stepwise_analysis.py# Layer-wise & scale-wise feature analysis
│       ├── csd100_pairs.py     # CSD100 benchmark case loader
│       ├── csd100_evaluator.py # CSD100 evaluation pipeline
│       └── plotting.py         # Visualization and grid export utilities
├── scripts/                    # Production CLIs & reproduction runners
│   ├── download_weights.py     # Download Infinity-2B checkpoints & VAE
│   ├── infer_style_transfer.py # Single-image PFB + SAC style generation
│   ├── infer_incremental.py    # Incremental generation with auto-naming
│   ├── eval_paper_teaser.py    # Reproduce teaser 3×6 evaluation cases
│   ├── eval_paper_batch.py     # Batch evaluation runner for paper styles
│   ├── eval_10x5_grid.py       # 10 prompts × 5 styles evaluation matrix
│   ├── eval_stochastic_sampling.py # Stochasticity temperature & jitter runner
│   ├── compute_clip_metrics.py # Compute CLIP style & text alignment scores
│   └── generate_zoo_showcase.py# Generate multi-seed assets for web exhibition
└── README.md
```

---

## 🚀 Quickstart

### 1. Environment Setup

Create and activate a Python 3.10 environment:

```bash
conda create -n chronosart python=3.10 -y
conda activate chronosart

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r third-party/Infinity/requirements.txt
pip install huggingface_hub hf-transfer ftfy regex rembg
```

Link the local packages into your environment:

```bash
python3 -c "import site, pathlib; p = pathlib.Path(site.getsitepackages()[0]) / 'chronosart.pth'; p.write_text(str(pathlib.Path('src').resolve()) + '\n' + str(pathlib.Path('third-party/Infinity').resolve()) + '\n')"
```

### 2. Download Checkpoints

Download official **Infinity-2B**, **d32reg VAE**, and **FLAN-T5-XL** text encoder weights:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python3 scripts/download_weights.py --output-dir weights
```

---

## 🎨 Running Inference

### Single Image Style Personalization (PFB + SAC)

```bash
python3 scripts/infer_style_transfer.py \
  --prompt "A kettle" \
  --style-image styles/08_pencil_sketch.png \
  --output output/kettle_pencil.png
```

### Reproducing Paper Benchmark (Teaser 3×6 Cases)

```bash
python3 scripts/eval_paper_teaser.py \
  --weights-dir weights \
  --output-dir output/paper_teaser
```

### Evaluating Safe Stochastic Brushwork

```bash
python3 scripts/eval_stochastic_sampling.py \
  --prompt "A noble lion with a majestic mane" \
  --style-image styles/08_pencil_sketch.png \
  --seeds 42 108 999 \
  --output-dir output/lion_stochastic
```

---

## 🌐 Interactive Wildlife Camera Roll

The project includes an interactive whiteboard web application inspired by camera roll brainstorm boards:
- **Pannable Whiteboard Canvas:** Freely drag and pan across 10 wild animal species.
- **Layered Polaroid Stacks & Pastel Notes:** Realistic 3-layer photo piles with handwritten field notes.
- **Live Stochastic Animation:** Click any snapshot to zoom into the stage and watch real-time stop-motion brushwork cycling between seeds.
- **Dual Exhibition:** Instant switching between **Pencil Sketch** and **Colorful Origami**.

Run the local server:
```bash
python3 -m http.server 8088 --directory docs
```
Visit `http://localhost:8088/` in your browser.

---

## 📜 Acknowledgements

This project implements and builds upon:
- **Paper:** *"A Training-Free Style-Personalization via SVD-Based Feature Decomposition"*
- **Base Model:** [FoundationVision Infinity](https://github.com/FoundationVision/Infinity) (Visual Autoregressive Modeling)
- **Text Conditioning:** Google [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl)
