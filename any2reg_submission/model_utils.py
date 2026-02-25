"""
Shared utilities for warping and CTE template. Self-contained (no external codebase).
"""
from typing import Optional

import torch
import torch.nn.functional as F


def warp_2d(
    images: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp 2D images using displacement field. flow: (..., 2, H, W) dx, dy."""
    if images.dim() == 5:
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        flow_flat = flow.view(B * N, 2, H, W) if flow.dim() == 5 else flow
        warped_flat = _warp_2d_single(images_flat, flow_flat, mode)
        return warped_flat.view(B, N, C, H, W)
    return _warp_2d_single(images, flow, mode)


def _warp_2d_single(
    images: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Warp 2D to match MIR SpatialTransformer (VoxelMorph) used in training.
    flow: (N, 2, H, W) with flow[:, 0] = dy (row), flow[:, 1] = dx (col).
    Uses same normalization and align_corners=False as MIR.
    """
    flow = flow.to(images.device)
    N, C, H, W = images.shape
    device, dtype = images.device, images.dtype
    # Grid: row (y), col (x) — same as MIR vectors = [arange(0,H), arange(0,W)]
    yy = torch.arange(H, device=device, dtype=dtype)
    xx = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    yy = yy.unsqueeze(0).expand(N, -1, -1)
    xx = xx.unsqueeze(0).expand(N, -1, -1)
    # MIR: new_locs = grid + flow, flow[:,0]=dy, flow[:,1]=dx
    row_new = yy + flow[:, 0]
    col_new = xx + flow[:, 1]
    # MIR normalize: 2 * (coord / (shape[i]-1) - 0.5)
    row_norm = 2.0 * (row_new / max(H - 1, 1) - 0.5)
    col_norm = 2.0 * (col_new / max(W - 1, 1) - 0.5)
    # grid_sample expects (x, y) = (col, row)
    grid = torch.stack((col_norm, row_norm), dim=-1)
    return F.grid_sample(images, grid, mode=mode, padding_mode="border", align_corners=False)


def compose_flow(base_flow: torch.Tensor, residual_flow: torch.Tensor) -> torch.Tensor:
    """Compose displacement fields: base_flow + warp(residual_flow, base_flow)."""
    assert base_flow.shape == residual_flow.shape
    warped_residual = warp_2d(residual_flow, base_flow)
    return base_flow + warped_residual


def compute_cte_template(
    images: torch.Tensor,
    downsample: int = 1,
) -> torch.Tensor:
    """PCA-weighted template from (N, C, H, W). Returns (1, C, H, W)."""
    assert images.dim() == 4, f"expected (N,C,H,W), got {images.shape}"
    N, C, H, W = images.shape
    assert N > 1, "CTE requires N > 1"
    if downsample > 1:
        images_for_corr = F.avg_pool2d(images, kernel_size=downsample, stride=downsample)
    else:
        images_for_corr = images
    flat = images_for_corr.reshape(N, -1)
    corr_mat = torch.corrcoef(flat)
    eigvals, eigvecs = torch.linalg.eigh(corr_mat)
    principal_vec = eigvecs[:, -1]
    weight_sum = principal_vec.sum()
    if torch.isclose(weight_sum.abs(), torch.tensor(0.0, device=principal_vec.device, dtype=principal_vec.dtype)):
        w = torch.full_like(principal_vec, 1.0 / N)
    else:
        w = principal_vec / weight_sum
    w = w.view(N, 1, 1, 1)
    template = (images * w).sum(dim=0, keepdim=True)
    return template


def compute_cte_template_batched(x: torch.Tensor) -> torch.Tensor:
    """CTE template for (B, N, C, H, W) -> (B, C, H, W)."""
    B, N, C, H, W = x.shape
    assert N > 1
    flat = x.view(B, N, -1)
    flat_centered = flat - flat.mean(dim=2, keepdim=True)
    std = (flat_centered.pow(2).sum(dim=2, keepdim=True) + 1e-12).sqrt()
    flat_norm = flat_centered / std
    corr = torch.bmm(flat_norm, flat_norm.transpose(1, 2)) / flat_norm.shape[2]
    _eigvals, eigvecs = torch.linalg.eigh(corr)
    w = eigvecs[:, :, -1]
    w_sum = w.sum(dim=1, keepdim=True)
    w = torch.where(
        torch.isclose(w_sum.abs(), torch.zeros_like(w_sum)),
        torch.full_like(w, 1.0 / N),
        w / w_sum,
    )
    w = w.view(B, N, 1, 1, 1)
    return (x * w).sum(dim=1)
