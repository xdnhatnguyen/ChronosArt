"""auto_research/candidate.py - The Single File the AI Agent Edits.

In the Karpathy AutoResearch protocol, this is the ONLY file the agent modifies.
Everything here is fair game:
- SVD decomposition math (singular value decay curves, energy thresholds, rank selection)
- Principal Feature Blending (PFB) injection rules and scale ranges
- Structural Attention Correction (SAC) parameters and attention head ratios
- Safe Stochastic schedules (scale-wise temperature curves, token jittering rates)

Goal: Maximize the composite score (composite_score) in evaluate.py.
Constraint: Keep it mathematically grounded, elegant, and avoid bloat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch


@dataclass
class CandidateConfig:
    """Hyperparameters and architectural knobs for Style Personalization.
    
    The agent is free to add, modify, or remove parameters here.
    """
    # 1. SVD & PFB Knobs
    rank: int = 8
    alpha: float = 0.85
    decay_mode: str = "exponential"  # "exponential" | "power_law" | "cosine" | "energy_threshold"
    power_p: float = 1.5
    energy_ratio: float = 0.90
    pfb_feature_indices: list[int] = field(default_factory=lambda: [1, 2, 3])
    style_strength: float = 1.4

    # 2. Structural Attention Correction (SAC) Knobs
    enable_sac: bool = True
    sac_prediction_start: int = 2
    sac_strength: float = 0.75
    sac_head_ratio: float = 0.70

    # 3. Geometric & Spectral Refinements
    use_geodesic: bool = True
    use_fourier_phase: bool = True
    fourier_scales: list[int] = field(default_factory=lambda: [5])
    fourier_kappa: float = 0.28

    # 4. Safe Stochasticity Knobs
    use_scale_scheduled_tau: bool = True
    tau_coarse: float = 0.15
    tau_fine_min: float = 0.85
    tau_fine_max: float = 1.25
    top_p_fine: float = 0.92

    use_style_token_jitter: bool = True
    jitter_start_step: int = 5
    jitter_dropout: float = 0.15


# Active configuration instance used by evaluate.py
CONFIG = CandidateConfig()


def compute_singular_weights(
    singular_values: torch.Tensor,
    rank: int,
    alpha: float,
    decay_mode: str = "exponential",
    power_p: float = 1.5,
    energy_ratio: float = 0.90,
) -> torch.Tensor:
    """Compute weighting factors for the top-k singular values.
    
    The agent can experiment with different decay formulations here:
    - Exponential: exp(-alpha * (i - 1))
    - Power-law: (1 + alpha * (i - 1)) ** (-power_p)
    - Cosine decay: cos(pi * (i - 1) / (2 * r))
    - Energy threshold: binary or soft mask keeping top energy_ratio variance
    """
    device = singular_values.device
    dtype = singular_values.dtype
    indices = torch.arange(rank, device=device, dtype=dtype)

    if decay_mode == "exponential":
        weights = torch.exp(-float(alpha) * indices)
    elif decay_mode == "power_law":
        weights = (1.0 + float(alpha) * indices).pow(-float(power_p))
    elif decay_mode == "cosine":
        denom = max(1.0, float(rank - 1))
        weights = torch.cos((math.pi * 0.5) * (indices / denom)).clamp_min(0.0)
    elif decay_mode == "energy_threshold":
        total_energy = singular_values.sum().clamp_min(1e-8)
        cum_energy = torch.cumsum(singular_values, dim=0) / total_energy
        weights = (cum_energy <= float(energy_ratio)).to(dtype)
        # Ensure at least the first component is active
        weights[0] = 1.0
    else:
        weights = torch.exp(-float(alpha) * indices)

    return weights


def truncated_svd_custom(
    feature: torch.Tensor,
    rank: int = 8,
    alpha: float = 0.85,
    decay_mode: str = "exponential",
) -> torch.Tensor:
    """Compute truncated SVD on cumulative feature representation [B, C, H, W] or [B, C, T, H, W].
    
    Phi_{r, alpha}(F) = sum_{i=1}^r w_i * sigma_i * u_i * v_i^T
    """
    original_dtype = feature.dtype
    orig_shape = feature.shape
    B, C = orig_shape[0], orig_shape[1]
    
    # Reshape to 2D matrices per batch item: [B, C, N]
    matrices = feature.detach().float().reshape(B, C, -1)
    outputs = []
    
    for matrix in matrices:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        used_rank = min(int(rank), s.numel())
        weights = compute_singular_weights(
            s[:used_rank],
            used_rank,
            alpha=alpha,
            decay_mode=decay_mode,
            power_p=CONFIG.power_p,
            energy_ratio=CONFIG.energy_ratio,
        )
        weighted_s = s[:used_rank] * weights
        reconstruction = (u[:, :used_rank] * weighted_s.unsqueeze(0)) @ vh[:used_rank]
        outputs.append(reconstruction.reshape(orig_shape[1:]))
        
    return torch.stack(outputs).to(dtype=original_dtype)


def compute_pfb_delta(
    generation_feature: torch.Tensor,
    style_feature: torch.Tensor,
    scale_idx: int,
) -> torch.Tensor:
    """Calculate the style delta Delta_{PFB} to add to generation features.
    
    Can be dynamically conditioned on scale_idx (coarse vs fine scales).
    """
    style_feature = style_feature.to(generation_feature)
    
    # Adaptive rank/alpha based on scale if desired
    rank = CONFIG.rank
    alpha = CONFIG.alpha
    
    sty_svd = truncated_svd_custom(style_feature, rank=rank, alpha=alpha, decay_mode=CONFIG.decay_mode)
    gen_svd = truncated_svd_custom(generation_feature, rank=rank, alpha=alpha, decay_mode=CONFIG.decay_mode)
    
    delta = sty_svd - gen_svd
    return delta * float(CONFIG.style_strength)


def get_inference_args(seed: int | None = None) -> dict:
    """Export configuration dictionary formatted for StyleTransferEngine.paper_dual_path_generate.
    
    evaluate.py passes these arguments into the generation engine.
    """
    return {
        "seed": seed,
        "pfb_feature_indices": CONFIG.pfb_feature_indices,
        "style_strength": CONFIG.style_strength,
        "style_decay": CONFIG.alpha,
        "rank": CONFIG.rank,
        "enable_sac": CONFIG.enable_sac,
        "sac_prediction_start": CONFIG.sac_prediction_start,
        "sac_strength": CONFIG.sac_strength,
        "sac_head_ratio": CONFIG.sac_head_ratio,
        "use_geodesic": CONFIG.use_geodesic,
        "use_fourier_phase": CONFIG.use_fourier_phase,
        "fourier_scales": CONFIG.fourier_scales,
        "fourier_kappa": CONFIG.fourier_kappa,
        "use_scale_scheduled_tau": CONFIG.use_scale_scheduled_tau,
        "tau_coarse": CONFIG.tau_coarse,
        "tau_fine_min": CONFIG.tau_fine_min,
        "tau_fine_max": CONFIG.tau_fine_max,
        "top_p_fine": CONFIG.top_p_fine,
        "use_style_token_jitter": CONFIG.use_style_token_jitter,
        "jitter_start_step": CONFIG.jitter_start_step,
        "jitter_dropout": CONFIG.jitter_dropout,
    }
