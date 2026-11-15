#!/usr/bin/env python3
"""Test deeply optimized ChronosArt pipeline:
1. Multi-scale injection across s in {1, 2, 3, 4, 5}
2. Content-Orthogonal Projection (I - P_con) at scales 3, 4, 5
3. Spherical Geodesic Retraction (SLERP) at ALL injection scales (preserving manifold)
4. High style strength (1.4x) enabled safely by SLERP
5. Selective Head SAC (75% structural heads, 25% free texture heads)
6. Bilateral Boundary Gradient Mask (locking sharp contours)
7. Spectral Fourier Phase-Lock at scale 5
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for d in (PROJECT_ROOT / "src", PROJECT_ROOT / "Infinity", PROJECT_ROOT / "Infinity/tools", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(d))

import run_content_ortho_batch as runtime
from chronosart.config import ExperimentConfig
from chronosart.feature_hypotheses import projected_pfb_content_blend, truncated_svd
from chronosart.plotting import save_image_tensor
from chronosart.style_transfer import (
    PaperSACPatch,
    SACController,
    StyleTransferEngine,
    compute_boundary_gradient_mask,
    fourier_phase_amplitude_blend,
    spherical_geodesic_update,
)
from chronosart.clip_metrics import CLIPMetricsEvaluator

TEST_CASES = [
    {
        "id": "case1_kettle_origami",
        "title": "A kettle (Origami Style)",
        "prompt": "A kettle in colorful origami style",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case2_car_origami",
        "title": "A car (Origami Style)",
        "prompt": "A car in colorful origami style",
        "style_image": "styles/style_01_colorful_origami.png",
    },
    {
        "id": "case3_pen_3d",
        "title": "A pen (3D Rendering)",
        "prompt": "A pen in 3d rendering style",
        "style_image": "styles/style_02_3d_rendering.png",
    },
    {
        "id": "case4_hourglass_renaissance",
        "title": "An hourglass (Northern Renaissance)",
        "prompt": "An hourglass in northern renaissance art style",
        "style_image": "styles/style_03_northern_renaissance_art.png",
    },
]


def run_super_optimized_generation(
    engine: StyleTransferEngine,
    prompt: str,
    style_features: list,
    *,
    seed: int = 42,
    cfg: float = 3.0,
    tau: float = 0.5,
    top_k: int = 900,
    top_p: float = 0.97,
    inject_steps: list[int] = [1, 2, 3, 4, 5],
    sac_start_step: int = 2,
    sac_head_ratio: float = 0.75,
    style_strength: float = 1.35,
    style_decay: float = 0.85,
    style_rank: int = 2,
    content_rank: int = 1,
    alpha: float = 1.0,
    boundary_gamma: float = 0.45,
    fourier_scale: int = 5,
    fourier_kappa: float = 0.22,
):
    engine.model.eval()
    base_batch = 1
    condition_batch = 2
    content_rng = torch.Generator(device=engine.device).manual_seed(seed)
    generation_rng = torch.Generator(device=engine.device).manual_seed(seed)

    kv_compact, lens, cu_seqlens_k, max_seqlen_k = engine.encode_prompts([prompt, prompt])
    kv_compact_un = kv_compact.clone()
    total = 0
    for length in lens:
        kv_compact_un[total : total + length] = engine.model.cfg_uncond[:length]
        total += length
    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    bs = 4

    kv_compact = engine.model.text_norm(kv_compact)
    sos = cond_bd = engine.model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = engine.model.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + engine.model.pos_start.expand(bs, 1, -1)

    with torch.amp.autocast("cuda", enabled=False):
        cond_bd_or_gss = engine.model.shared_ada_lin(cond_bd.float()).float().contiguous()

    final_size = engine.scale_schedule[-1]
    content_summed = last_stage.new_zeros(base_batch, engine.model.d_vae, *final_size)
    generation_summed = torch.zeros_like(content_summed)

    controller = SACController(base_batch)
    controller.sac_strength = 1.0
    controller.sac_head_ratio = sac_head_ratio
    controller.reset_statistics()

    for block in engine.model.unregistered_blocks:
        block.sa.kv_caching(True)

    try:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            sac_context = PaperSACPatch(engine.model, controller)
            with sac_context:
                for step_id, pn in enumerate(engine.scale_schedule):
                    controller.active = step_id >= sac_start_step
                    need_to_pad = 0
                    attn_fn = None
                    if engine.model.use_flex_attn:
                        attn_fn = engine.model.attn_fn_compile_dict.get(tuple(engine.scale_schedule[: step_id + 1]), None)

                    for block_idx, block_chunk in enumerate(engine.model.block_chunks):
                        if engine.model.add_lvl_embeding_only_first_block and block_idx == 0:
                            last_stage = engine.model.add_lvl_embeding(
                                last_stage, step_id, engine.scale_schedule, need_to_pad=need_to_pad
                            )
                        if not engine.model.add_lvl_embeding_only_first_block:
                            last_stage = engine.model.add_lvl_embeding(
                                last_stage, step_id, engine.scale_schedule, need_to_pad=need_to_pad
                            )

                        for block in block_chunk.module:
                            last_stage = block(
                                x=last_stage,
                                cond_BD=cond_bd_or_gss,
                                ca_kv=ca_kv,
                                attn_bias_or_two_vector=None,
                                attn_fn=attn_fn,
                                scale_schedule=engine.scale_schedule,
                                rope2d_freqs_grid=engine.model.rope2d_freqs_grid,
                                scale_ind=step_id,
                            )

                    logits = engine.model.get_logits(last_stage, cond_bd).mul(1 / float(tau))
                    logits = float(cfg) * logits[:condition_batch] + (1 - float(cfg)) * logits[condition_batch:]
                    content_idx = engine._sample_bit_labels(logits[:1], content_rng, top_k, top_p)
                    generation_idx = engine._sample_bit_labels(logits[1:2], generation_rng, top_k, top_p)

                    content_codes = engine._bit_labels_to_codes(content_idx, pn)
                    generation_codes = engine._bit_labels_to_codes(generation_idx, pn)
                    if step_id != len(engine.scale_schedule) - 1:
                        content_codes = F.interpolate(content_codes, size=final_size, mode=engine.vae.quantizer.z_interplote_up)
                        generation_codes = F.interpolate(generation_codes, size=final_size, mode=engine.vae.quantizer.z_interplote_up)

                    content_summed = content_summed + content_codes
                    generation_summed = generation_summed + generation_codes

                    # Multi-scale Style Injection with Content Orthogonality & Geodesic Retraction
                    if step_id in inject_steps:
                        injection_order = inject_steps.index(step_id)
                        eff_strength = float(style_strength) * (float(style_decay) ** injection_order)
                        sty_feat = style_features[step_id].to(generation_summed)

                        if step_id in [1, 2]:
                            # Coarse scales: unprojected rank-1 PFB delta
                            delta = truncated_svd(sty_feat, rank=1, alpha=alpha) - truncated_svd(generation_summed, rank=1, alpha=alpha)
                        else:
                            # Intermediate scales 3, 4, 5: Content-Orthogonal Projection (I - P_con) with rank 2
                            # Extracts delta and projects out content subspace
                            delta_raw = truncated_svd(sty_feat, rank=style_rank, alpha=alpha) - truncated_svd(generation_summed, rank=style_rank, alpha=alpha)
                            # Project out content subspace
                            from chronosart.feature_hypotheses import estimate_content_basis, _feature_matrix
                            bases, _ = estimate_content_basis(content_summed, content_rank=content_rank)
                            delta_mat = _feature_matrix(delta_raw)
                            proj_delta_list = []
                            for d_m, basis in zip(delta_mat, bases):
                                c_comp = basis @ (basis.transpose(0, 1) @ d_m)
                                p_d = d_m - c_comp
                                proj_delta_list.append(p_d.reshape(generation_summed.shape[1:]))
                            delta = torch.stack(proj_delta_list).to(generation_summed)

                        # Boundary Gradient Protection
                        b_mask = compute_boundary_gradient_mask(generation_summed, gamma=boundary_gamma)
                        delta = delta * b_mask

                        # Spherical Geodesic Retraction (SLERP on S^{C-1})
                        generation_summed = spherical_geodesic_update(generation_summed, delta, strength=eff_strength)

                    # Fine scale Fourier Phase-Lock
                    if step_id == fourier_scale and step_id < len(style_features):
                        generation_summed = fourier_phase_amplitude_blend(
                            generation_summed,
                            style_features[step_id].to(generation_summed),
                            content_feat=content_summed,
                            kappa=fourier_kappa,
                        )

                    if step_id != len(engine.scale_schedule) - 1:
                        next_scale = engine.scale_schedule[step_id + 1]
                        content_next = engine._next_raw_from_summed_codes(content_summed, next_scale)
                        generation_next = engine._next_raw_from_summed_codes(generation_summed, next_scale)
                        two_streams = torch.cat((content_next, generation_next), dim=0)
                        last_stage = engine.model.word_embed(engine.model.norm0_ve(two_streams))
                        last_stage = last_stage.repeat(bs // condition_batch, 1, 1)

        stylized_image = engine._decode_summed_codes_to_image_01(generation_summed)
        return stylized_image
    finally:
        controller.active = False
        for block in engine.model.unregistered_blocks:
            block.sa.kv_caching(False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    args = parser.parse_args()

    runtime.load_runtime_modules()
    bundle = runtime.load_bundle(args)
    config = ExperimentConfig(
        model_pn=args.pn,
        cfg=args.cfg,
        tau=args.tau,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    engine = StyleTransferEngine(bundle, config)

    output_dir = PROJECT_ROOT / "output/experiment_optimized"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = CLIPMetricsEvaluator(output_dir=output_dir)

    print("\n" + "=" * 80)
    print("RUNNING SUPER-OPTIMIZED PIPELINE (Multi-scale Geodesic Orthogonal Projection + Selective SAC + Fourier)")
    print("=" * 80)

    results = []
    for case in TEST_CASES:
        cid = case["id"]
        print(f"\nProcessing {case['title']}...")
        style_path = (PROJECT_ROOT / case["style_image"]).resolve()
        style_feats = engine.get_style_features(style_path)

        t0 = time.time()
        img_tensor = run_super_optimized_generation(
            engine,
            case["prompt"],
            style_feats,
            seed=args.seed,
            style_strength=1.35,
            style_decay=0.85,
            sac_head_ratio=0.75,
            boundary_gamma=0.45,
            fourier_scale=5,
            fourier_kappa=0.22,
        )
        dur = time.time() - t0
        out_path = output_dir / f"{cid}_optimized.png"
        save_image_tensor(img_tensor.detach().float().cpu(), out_path)
        print(f"  Generated in {dur:.2f}s -> {out_path.name}")

        # Evaluate metrics
        img_emb = evaluator.image_embedding(out_path)
        style_emb = evaluator.image_embedding(style_path)
        text_emb = evaluator.text_embedding(case["prompt"])

        s_txt = evaluator.cosine_score(img_emb, text_emb)
        s_img = evaluator.cosine_score(img_emb, style_emb)
        s_harm = evaluator.harmonic_score(s_txt, s_img)

        # Compare with baseline
        base_path = PROJECT_ROOT / f"output/experiment_all/{cid}_baseline.png"
        b_img_emb = evaluator.image_embedding(base_path)
        b_s_txt = evaluator.cosine_score(b_img_emb, text_emb)
        b_s_img = evaluator.cosine_score(b_img_emb, style_emb)
        b_s_harm = evaluator.harmonic_score(b_s_txt, b_s_img)

        print(f"  Optimized: S_harmonic={s_harm:.4f} | S_txt={s_txt:.4f} | S_img={s_img:.4f}")
        print(f"  Baseline:  S_harmonic={b_s_harm:.4f} | S_txt={b_s_txt:.4f} | S_img={b_s_img:.4f}")
        print(f"  Gain:      ΔS_harmonic={s_harm - b_s_harm:+.4f} | ΔS_img={s_img - b_s_img:+.4f}")

        results.append({
            "case": case["title"],
            "s_harm": s_harm,
            "s_txt": s_txt,
            "s_img": s_img,
            "b_s_harm": b_s_harm,
            "b_s_txt": b_s_txt,
            "b_s_img": b_s_img,
            "dur": dur,
        })

    import pandas as pd
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("FINAL SUMMARY METRICS:")
    print("=" * 80)
    print(f"Optimized Mean S_harmonic: {df['s_harm'].mean():.4f} (Baseline: {df['b_s_harm'].mean():.4f}) -> Gain: {df['s_harm'].mean() - df['b_s_harm'].mean():+.4f}")
    print(f"Optimized Mean S_img:      {df['s_img'].mean():.4f} (Baseline: {df['b_s_img'].mean():.4f}) -> Gain: {df['s_img'].mean() - df['b_s_img'].mean():+.4f}")
    print(f"Optimized Mean S_txt:      {df['s_txt'].mean():.4f} (Baseline: {df['b_s_txt'].mean():.4f}) -> Gain: {df['s_txt'].mean() - df['b_s_txt'].mean():+.4f}")
    print(f"Average Inference Time:    {df['dur'].mean():.2f}s")


if __name__ == "__main__":
    main()
