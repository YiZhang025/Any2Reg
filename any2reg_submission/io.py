"""
Minimal I/O utilities for loading cardiac MRI data and features.

This module provides basic loading functionality for NIfTI images and
pre-computed feature maps. It is intentionally minimal and does not include
comprehensive error handling or production-grade validation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import nibabel as nib
import numpy as np
import torch


def _clip_intensity(img: np.ndarray, percentile: float) -> np.ndarray:
    """
    Clip intensity at upper percentile (no lower clip).
    So that normalization is not dominated by a few bright outliers.
    """
    p = max(0.0, min(100.0, float(percentile)))
    clip_max = np.percentile(img, p)
    return np.clip(img, a_min=None, a_max=clip_max)


def load_nifti_case(
    nifti_path: Path,
    center_crop_size: Tuple[int, int] = (192, 192),
    normalize: bool = True,
    intensity_clip_percentile: Optional[float] = 97.0,
) -> Dict[str, Any]:
    """
    Load a single NIfTI case (4D: H×W×T or 3D sequence).
    
    Args:
        nifti_path: Path to .nii.gz file
        center_crop_size: Target (H, W) for center crop/pad
        normalize: Whether to normalize to [0, 1]
        intensity_clip_percentile: If not None, clip intensities at this percentile
            (upper bound) before normalizing. Set to None to disable.
    
    Returns:
        Dictionary containing:
            - 'images': torch.Tensor (T, 1, H, W) preprocessed
            - 'images_original': torch.Tensor (T, 1, H_orig, W_orig) raw
            - 'affine': numpy array, NIfTI affine
            - 'shape_hwt': tuple (H_orig, W_orig, T)
            - 'case_id': str, filename
    
    Note:
        Preprocessing: center crop, optional percentile clip, then normalize to [0, 1].
    """
    nim = nib.load(str(nifti_path))
    data = np.asanyarray(nim.dataobj).astype(np.float32)
    
    # Assume (H, W, T) format
    if data.ndim == 2:
        data = data[..., np.newaxis]
    assert data.ndim == 3, f"Expected 3D data, got shape {data.shape}"
    
    H, W, T = data.shape
    
    # Original images (no crop, no normalization)
    images_original = torch.from_numpy(data.copy()).permute(2, 0, 1).unsqueeze(1).float()
    
    # Preprocessed: center crop, optional intensity clip, then normalize to [0,1]
    data_crop = _center_crop_pad_hw(data, center_crop_size)
    if normalize:
        if intensity_clip_percentile is not None:
            data_crop = _clip_intensity(data_crop, intensity_clip_percentile)
        data_crop = data_crop / (np.max(data_crop) + 1e-6)
    images = torch.from_numpy(data_crop).permute(2, 0, 1).unsqueeze(1).float()
    
    return {
        'images': images,
        'images_original': images_original,
        'affine': nim.affine.copy(),
        'shape_hwt': (H, W, T),
        'case_id': nifti_path.name,
    }


def load_feature_map(
    feature_path: Path,
    feature_key: str = "logits_final",
    center_crop_size: Tuple[int, int] = (192, 192),
) -> Optional[torch.Tensor]:
    """
    Load pre-computed feature map from .npz file.
    
    Args:
        feature_path: Path to .npz file containing features
        feature_key: Key name in npz file (default: 'logits_final')
        center_crop_size: Target (H, W) for center crop/pad
    
    Returns:
        torch.Tensor (T, 1, H, W) or None if loading fails
    
    Note:
        Assumes feature format (C, T, H, W) and averages over channel dimension.
        This is a simplified version suitable for demonstration purposes.
    """
    if not feature_path.exists():
        return None
    
    try:
        data = np.load(feature_path)
        if feature_key not in data:
            return None
        
        feature = data[feature_key].astype(np.float32)  # (C, T, H, W)
        
        # Center crop/pad
        feature = _center_crop_or_pad(feature, center_crop_size)
        
        # Average over channel dimension
        feature = feature.mean(axis=0)  # (T, H, W)
        
        # Transpose to (H, W, T) then to (T, 1, H, W)
        feature = torch.from_numpy(np.transpose(feature, (1, 2, 0))).float()
        feature = feature.permute(2, 0, 1).unsqueeze(1)
        
        # Normalize
        feature = feature.transpose(2, 3)
        max_val = feature.max()
        if max_val > 0:
            feature = feature / max_val
        
        return feature
    except Exception as e:
        print(f"Warning: Failed to load feature from {feature_path}: {e}")
        return None


def save_nifti(
    data: torch.Tensor,
    affine: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save tensor as NIfTI file.
    
    Args:
        data: torch.Tensor, expected shape (T, 1, H, W)
        affine: NIfTI affine matrix
        output_path: Output file path
    
    Note:
        This function performs minimal validation. For production use,
        additional checks would be required.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert (T, 1, H, W) -> (H, W, T)
    if data.dim() == 4 and data.shape[1] == 1:
        arr = data.squeeze(1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    else:
        arr = data.cpu().numpy().astype(np.float32)
    
    nim = nib.Nifti1Image(arr, affine)
    nib.save(nim, str(output_path))


def _center_crop_pad_hw(
    img: np.ndarray,
    crop_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Center crop or pad (H,W,T) image to target size.
    
    Internal helper: center crop or pad to target size.
    """
    H, W, T = img.shape
    crop_h, crop_w = crop_hw
    
    start_h = max(0, (H - crop_h) // 2)
    start_w = max(0, (W - crop_w) // 2)
    end_h = min(H, start_h + crop_h)
    end_w = min(W, start_w + crop_w)
    
    src_h, src_w = end_h - start_h, end_w - start_w
    dst_begin_h = (crop_h - src_h) // 2
    dst_begin_w = (crop_w - src_w) // 2
    
    out = np.zeros((crop_h, crop_w, T), dtype=img.dtype)
    out[dst_begin_h:dst_begin_h + src_h, dst_begin_w:dst_begin_w + src_w, :] = (
        img[start_h:end_h, start_w:end_w, :]
    )
    return out


def _center_crop_or_pad(
    img: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Center crop or pad last two dimensions to target size.
    
    Internal helper for feature map processing.
    """
    th, tw = target_size
    h, w = img.shape[-2:]
    
    if h >= th and w >= tw:
        top = (h - th) // 2
        left = (w - tw) // 2
        return img[..., top:top + th, left:left + tw]
    
    # Need padding
    cropped = img[..., max(0, (h - th) // 2):max(0, (h - th) // 2) + min(th, h),
                       max(0, (w - tw) // 2):max(0, (w - tw) // 2) + min(tw, w)]
    
    pad_h_before = (th - cropped.shape[-2]) // 2
    pad_h_after = th - cropped.shape[-2] - pad_h_before
    pad_w_before = (tw - cropped.shape[-1]) // 2
    pad_w_after = tw - cropped.shape[-1] - pad_w_before
    
    pad_width = [(0, 0)] * (cropped.ndim - 2) + [
        (pad_h_before, pad_h_after),
        (pad_w_before, pad_w_after),
    ]
    return np.pad(cropped, pad_width, mode='constant', constant_values=0)


def resolve_stone_case(stone_root: Path, subject_token: str = "138_4") -> Optional[Path]:
    """
    Find STONE case matching subject token.
    
    Args:
        stone_root: Path to STONE data root (contains data/ subdirectory)
        subject_token: Subject identifier (e.g., "138_4")
    
    Returns:
        Path to matched .nii.gz file or None if not found
    
    Note:
        This performs simple glob matching. Production code would require
        more robust case resolution and validation.
    """
    data_dir = stone_root / "data"
    if not data_dir.exists():
        return None
    
    # Look for files containing subject token
    candidates = list(data_dir.glob(f"*{subject_token}*.nii.gz"))
    if not candidates:
        return None
    
    # Return first match (assumes unique or acceptable first match)
    return candidates[0]


def resolve_acdc_case(acdc_root: Path, select_mid_slice: bool = True) -> Optional[Path]:
    """
    Find ACDC case from first subject, optionally selecting mid-slice.
    
    Args:
        acdc_root: Path to ACDC data root (contains data/ subdirectory)
        select_mid_slice: If True, attempt to find mid-slice case
    
    Returns:
        Path to selected .nii.gz file or None if not found
    
    Note:
        This uses simple lexical ordering and assumes standard ACDC naming.
        For robust production use, explicit case lists would be preferred.
    """
    data_dir = acdc_root / "data"
    if not data_dir.exists():
        return None
    
    all_cases = sorted(data_dir.glob("*.nii.gz"))
    if not all_cases:
        return None
    
    # Group by subject (assumes format patient###_...)
    first_subject = all_cases[0].name.split('_')[0]
    subject_cases = [c for c in all_cases if c.name.startswith(first_subject)]
    
    if not subject_cases:
        return all_cases[0]
    
    if select_mid_slice and len(subject_cases) > 1:
        # Select middle index as proxy for mid-slice
        mid_idx = len(subject_cases) // 2
        return subject_cases[mid_idx]
    
    return subject_cases[0]
