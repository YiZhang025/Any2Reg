"""
Minimal inference implementation for Any2RegNet and baseline methods.
Self-contained: uses local model and model_utils only.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import torch

from .model_utils import compute_cte_template, warp_2d


def run_raw_baseline(
    images: torch.Tensor,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """
    Run raw baseline (zero displacement).
    
    This is a reference method that applies no registration. It is useful
    for visualization and as a sanity check baseline, but is not competitive
    as a registration method.
    
    Args:
        images: (T, 1, H, W) input images
        device: Computation device
    
    Returns:
        Dictionary containing:
            - 'disp': (T, 2, H, W) displacement field (zeros)
            - 'warped': (T, 1, H, W) warped images (identical to input)
            - 'template': (1, 1, H, W) mean template
    
    Note:
        This method is labeled as reference-only in documentation and
        outputs. It should not be compared directly with learned methods.
    """
    images = images.to(device)
    T, C, H, W = images.shape
    
    # Zero displacement
    disp = torch.zeros((T, 2, H, W), dtype=images.dtype, device=device)
    
    # "Warped" images are just original
    warped = images.clone()
    
    # Template is simple mean
    template = warped.mean(dim=0, keepdim=True)
    
    return {
        'disp': disp,
        'warped': warped,
        'template': template,
    }


def _run_any2regnet_real(
    images: torch.Tensor,
    feature_maps: Optional[torch.Tensor],
    checkpoint_path: Path,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load Any2RegNet from this package and run inference. Returns None on failure."""
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        return None

    from .model import create_any2regnet

    images = images.to(device)
    if feature_maps is not None:
        feature_maps = feature_maps.to(device)
    else:
        print("  ⚠ Any2RegNet (raw+logits) checkpoint used without feature_maps; set feature_dir for correct input.")

    T, C, H, W = images.shape
    feat_in_channels = int(feature_maps.shape[1]) if feature_maps is not None and feature_maps.dim() == 4 else 1

    try:
        model = create_any2regnet(
            sample_images=images,
            in_channels=1,
            enc_channels=16,
            dec_channels=16,
            num_iters=1,
            feat_in_channels=feat_in_channels,
            fusion_mode="weighted_add",
            encoder_aggregation="cte",
            use_checkpoint=False,
        ).to(device)
    except Exception:
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception:
        return None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    with torch.no_grad():
        warped, flow, template = model(images, feature_maps=feature_maps)

    return {
        "disp": flow,
        "warped": warped,
        "template": template,
    }


def run_any2regnet_inference(
    images: torch.Tensor,
    feature_maps: Optional[torch.Tensor],
    checkpoint_path: Optional[Path],
    device: torch.device = torch.device('cpu'),
    num_iters: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Run Any2RegNet inference.
    
    When the full model is available and checkpoint_path is a valid file,
    loads the real Any2RegNet and runs with (images, feature_maps). The
    included checkpoint (raw_logits_best) is trained on raw+logits; set
    feature_dir and provide precomputed features for correct input.
    
    Otherwise falls back to a stub that ignores feature_maps (output not meaningful).
    
    Args:
        images: (T, 1, H, W) input images
        feature_maps: Optional (T, C, H, W) pre-computed features (e.g. logits); required for correct results with raw_logits checkpoint
        checkpoint_path: Path to pretrained weights
        device: Computation device
        num_iters: Number of refinement iterations (used by real model from checkpoint)
    
    Returns:
        Dictionary: 'disp', 'warped', 'template'; may include '_stub_warning' if stub was used.
    """
    # Try real model first (checkpoint + optional external model)
    if checkpoint_path:
        out = _run_any2regnet_real(images, feature_maps, Path(checkpoint_path), device)
        if out is not None:
            return out

    # Fallback: stub (ignores feature_maps and checkpoint)
    print("WARNING: Using stub Any2RegNet implementation (feature_maps and checkpoint ignored).")
    print("         For real inference: provide checkpoint and feature_dir.")
    images = images.to(device)
    T, C, H, W = images.shape
    disp = torch.zeros((T, 2, H, W), dtype=images.dtype, device=device)
    for _ in range(num_iters):
        warped = warp_2d(images, disp)
        template = compute_cte_template(warped)
        diff = images - template
        disp = disp - 0.1 * torch.stack([diff.squeeze(1), diff.squeeze(1)], dim=1)
    warped = warp_2d(images, disp)
    template = compute_cte_template(warped)
    return {
        "disp": disp,
        "warped": warped,
        "template": template,
        "_stub_warning": "This output is from stub implementation",
    }


def load_any2regnet_model_stub(checkpoint_path: Path, device: torch.device):
    """Legacy stub; real model is loaded in _run_any2regnet_real via create_any2regnet."""
    return None
