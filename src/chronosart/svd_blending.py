"""SVD-Based Feature Decomposition and Principal Feature Blending (PFB).

Implements the core mathematical formulation from:
"A Training-Free Style-Personalization via SVD-Based Feature Decomposition"

All functions operate on cumulative VAE feature representations shaped ``[B, C, T, H, W]``
or ``[B, C, H, W]``.
"""

from __future__ import annotations

import torch


def _feature_matrix(feature: torch.Tensor) -> torch.Tensor:
    return feature.detach().float().reshape(feature.shape[0], feature.shape[1], -1)


def truncated_svd(feature: torch.Tensor, *, rank: int | None = None, alpha: float = 1.0) -> torch.Tensor:
    """Return the exponentially weighted low-rank reconstruction of a feature.
    
    Implements Eq. (5) from the paper:
    Phi_{r, alpha}(F) = sum_{i=1}^r exp(-alpha * (i - 1)) * sigma_i * u_i * v_i^T
    """
    original_dtype = feature.dtype
    matrices = _feature_matrix(feature)
    outputs = []
    for matrix in matrices:
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        used_rank = singular_values.numel() if rank is None else min(int(rank), singular_values.numel())
        weights = torch.exp(-float(alpha) * torch.arange(used_rank, device=matrix.device, dtype=matrix.dtype))
        reconstruction = (u[:, :used_rank] * (singular_values[:used_rank] * weights).unsqueeze(0)) @ vh[:used_rank]
        outputs.append(reconstruction.reshape(feature.shape[1:]))
    return torch.stack(outputs).to(dtype=original_dtype)


def principal_feature_blend(
    generation_feature: torch.Tensor,
    style_feature: torch.Tensor,
    *,
    rank: int | None = None,
    alpha: float = 1.0,
    strength: float = 1.0,
) -> torch.Tensor:
    """Replace the selected principal component of generation with style (PFB).
    
    Implements Eq. (4) from the paper:
    F_{s, PFB}^{gen} = F_s^{gen} + lambda * Delta_{PFB}
    where Delta_{PFB} = Phi_{r, alpha}(F_s^{sty}) - Phi_{r, alpha}(F_s^{gen})
    """
    if generation_feature.shape != style_feature.shape:
        raise ValueError(f"Feature shape mismatch: {generation_feature.shape} vs {style_feature.shape}")
    style_feature = style_feature.to(generation_feature)
    delta = truncated_svd(style_feature, rank=rank, alpha=alpha) - truncated_svd(
        generation_feature, rank=rank, alpha=alpha
    )
    return generation_feature + float(strength) * delta
