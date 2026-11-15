# ChronosArt

> **Training-Free Style Personalization with Visual Autoregressive Models**

ChronosArt implements high-fidelity, training-free style transfer on visual autoregressive transformers (specifically **Infinity-2B**). By extracting multiscale cumulative features, constructing principal style residuals, and projecting away content-aligned components, ChronosArt seamlessly transfers texture and artistic appearance from a reference style image while strictly preserving content structure, geometry, and composition.

---

## Method Overview

ChronosArt operates concurrently on a clean content stream and an editable stylized stream sharing the same prompt conditioning:

1. **Multiscale Feature Extraction:** Extracts cumulative VAE representations across autoregressive resolution scales ($1\times 1 \to 32\times 32$).
2. **Principal Feature Blending (PFB):** Derives a low-rank style residual via Singular Value Decomposition (SVD) between reference style features and generation features.
3. **Content Subspace Projection:** Estimates an empirical low-rank content basis from the clean stream and removes residual components aligned with this basis, preventing content corruption and semantic bleeding.
4. **Scale-Aware Geometric Decay:** Injects the purified style signal with decaying strength across coarse-to-fine scales.
5. **Structural Attention Correction (SAC):** Enforces query-key preservation in self-attention layers to anchor geometric alignment throughout autoregressive refinement.

---

## Repository Layout

```text
.
├── Infinity/            # Vendored FoundationVision Infinity engine
├── src/
│   └── chronosart/      # Core package: PFB, SAC, feature hypotheses, and metrics
├── scripts/             # Production CLIs for downloading, generation, and batch runs
│   ├── download_infinity_weights.py
│   ├── run_pfb_sac.py
│   ├── run_incremental_pfb_sac.py
│   ├── run_10x5_eval.py
│   ├── run_content_ortho_multiscale_decay_batch.py
│   ├── run_content_projection_batch.py
│   ├── run_original_pfb_sac_batch.py
│   └── evaluate_content_ortho_clip.py
├── prompts/             # Benchmark content prompts (Parti-190) and manifests
├── styles/              # Curated style references (FineStyle-21) and metadata
├── csd100/              # Step-wise analysis reference pairs
├── notebooks/           # Interactive exploration notebooks
└── tests/               # CPU verification test suite
```

---

## Quickstart

### 1. Environment Setup

Create and activate a Python 3.10 environment (e.g. conda environment `var`):

```bash
conda create -n var python=3.10 -y
conda activate var

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Infinity dependencies & utilities
pip install -r Infinity/requirements.txt
pip install huggingface_hub hf-transfer pytest
```

Link the source package into your environment:

```bash
python3 -c "import site, pathlib; p = pathlib.Path(site.getsitepackages()[0]) / 'chronosart.pth'; p.write_text(str(pathlib.Path('src').resolve()) + '\n' + str(pathlib.Path('Infinity').resolve()) + '\n')"
```

### 2. Download Model Checkpoints

Download official **Infinity-2B**, the compatible **d32reg VAE**, and **FLAN-T5-XL** tokenizer/encoder weights:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python3 scripts/download_infinity_weights.py --output-dir weights
```

---

## Running Inference

### Single Image Generation

Generate a stylized image using dual-path PFB + SAC:

```bash
python3 scripts/run_pfb_sac.py \
  --prompt "a cat" \
  --style-image styles/style_04_kid_crayon_drawing.png \
  --output output/test_01.png
```

To run sequential generations with automatically incrementing filenames (`output/test_01.png`, `test_02.png`, ...):

```bash
python3 scripts/run_incremental_pfb_sac.py \
  --prompt "a cat" \
  --style-image styles/style_04_kid_crayon_drawing.png
```

### Parallel Batch Evaluation (Multi-GPU)

Evaluate an $M \times N$ matrix of prompts and styles with multi-GPU sharding:

```bash
# GPU 0 (Shard 0)
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_10x5_eval.py --shard-index 0 --num-shards 2

# GPU 1 (Shard 1)
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_10x5_eval.py --shard-index 1 --num-shards 2
```

Outputs will be saved under `output/grid_10x5/` with generated images, clean content references, and tri-panel comparison strips.

### Content-Orthogonal Multiscale Paper Configuration

Run the complete content-projected multiscale decay schedule across the benchmark:

```bash
python3 scripts/run_content_ortho_multiscale_decay_batch.py
```

---

## Python API Usage

Intervene directly on latent VAE feature representations:

```python
import torch
from chronosart.feature_hypotheses import projected_pfb_content_blend

# generation_feature, style_feature, content_feature: [B, C, H, W]
edited_feature, diagnostics = projected_pfb_content_blend(
    generation_feature=gen_feat,
    style_feature=style_feat,
    content_feature=content_feat,
    style_rank=1,
    content_rank=1,
    projection_strength=1.0,
    preserve_mean=False,
    return_diagnostics=True,
)

print("Explained variance:", diagnostics["content_explained_variance"])
print("Removed fraction:", diagnostics["removed_fraction"])
```

---

## Verification & Tests

Run CPU unit tests to verify feature projection invariants, orthogonality preservation, and rank adaptivity:

```bash
python3 -m pytest -q tests
python3 -m compileall -q src scripts Infinity/infinity/models
```

---

## License

This project incorporates the Infinity framework under the Apache 2.0 license.
