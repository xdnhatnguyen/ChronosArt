"""Infinity runtime loader and bundle management.

Replaces legacy batch scripts with a clean, centralized loader for Infinity-2B,
FLAN-T5-XL, and visual tokenizers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFINITY_DIR = PROJECT_ROOT / "third-party" / "Infinity" if (PROJECT_ROOT / "third-party" / "Infinity").exists() else PROJECT_ROOT / "Infinity"
SRC_DIR = PROJECT_ROOT / "src"

for d in (str(SRC_DIR), str(INFINITY_DIR), str(INFINITY_DIR / "tools")):
    if d not in sys.path:
        sys.path.insert(0, d)

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer
from chronosart.stepwise_analysis import encode_prompts


def load_runtime_modules() -> None:
    """Compatibility hook to ensure dependencies are loaded."""
    pass


def scale_schedule(pn: str) -> list[tuple[int, int, int]]:
    aspect = min(h_div_w_templates, key=lambda value: abs(float(value) - 1.0))
    return [(1, h, w) for _, h, w in dynamic_resolution_h_w[aspect][pn]["scales"]]


class Bundle:
    def __init__(self, tokenizer, encoder, vae, model, schedule, image_size):
        self.device = "cuda"
        self.text_tokenizer = tokenizer
        self.text_encoder = encoder
        self.vae = vae
        self.infinity_model = model
        self.scale_schedule = schedule
        self.image_size = image_size


def load_bundle(args: argparse.Namespace) -> Bundle:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Infinity-2B inference")

    weights = getattr(args, "weights_dir", PROJECT_ROOT / "weights").resolve()
    required = {
        "model": weights / "infinity_2b_reg.pth",
        "vae": weights / "infinity_vae_d32reg.pth",
        "t5": weights / "flan-t5-xl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing weights:\n" + "\n".join(missing))

    pn = getattr(args, "pn", "0.25M")
    seed = getattr(args, "seed", 42)

    runtime_args = argparse.Namespace(
        pn=pn,
        model_path=str(required["model"]),
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=str(required["vae"]),
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_2b",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=str(required["t5"]),
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.0,
        use_flex_attn=0,
        cache_dir="/dev/shm",
        checkpoint_type="torch",
        seed=seed,
        bf16=1,
        enable_model_cache=0,
    )

    tokenizer, encoder = load_tokenizer(t5_path=runtime_args.text_encoder_ckpt)
    vae = load_visual_tokenizer(runtime_args)
    model = load_transformer(vae, runtime_args)
    if not hasattr(vae.quantizer, "lfq"):
        vae.quantizer.lfq = vae.quantizer.bsq

    image_size = {"0.06M": 256, "0.25M": 512, "0.60M": 768, "1M": 1024}[pn]
    return Bundle(tokenizer, encoder, vae, model, scale_schedule(pn), image_size)


def style_feature_at_step(bundle: Bundle, image_path: str, step: int) -> torch.Tensor:
    with torch.inference_mode():
        image = ImageOps.fit(
            Image.open(image_path).convert("RGB"),
            (bundle.image_size, bundle.image_size),
            Image.Resampling.LANCZOS,
        )
        image_m11 = to_tensor(image).unsqueeze(0).cuda().mul(2).sub(1)
        with torch.amp.autocast("cuda", enabled=False):
            _, _, _, indices, _, _ = bundle.vae.encode(image_m11.float(), scale_schedule=bundle.scale_schedule)
        return bundle.infinity_model.get_step_feature(indices, step)
