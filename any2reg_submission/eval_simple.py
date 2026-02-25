"""
Minimal evaluation metrics for sanity checking.

This module provides basic metrics only. It is NOT a comprehensive evaluation
suite and should not be used for rigorous performance comparison.

Scope:
- Simple Dice coefficient (if masks available)
- Basic displacement field statistics
- Reference-only markers for baseline methods

Not included:
- Statistical significance testing
- Multi-dataset aggregation
- Production-grade metric computation
"""

from typing import Dict, Optional

import numpy as np
import torch


def compute_dice_simple(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    label: int = 1,
) -> float:
    """
    Compute Dice coefficient for a single label between two masks.
    
    Args:
        mask_a: (H, W) integer mask
        mask_b: (H, W) integer mask
        label: Label value to compute Dice for
    
    Returns:
        Dice coefficient [0, 1]
    
    Note:
        This is a simplified implementation for demonstration. Production
        code would include edge case handling and validation.
    """
    a_binary = (mask_a == label).float()
    b_binary = (mask_b == label).float()
    
    intersection = (a_binary * b_binary).sum()
    union = a_binary.sum() + b_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()


def compute_pairwise_dice(
    masks: torch.Tensor,
    labels: list = [1, 2, 3],
) -> Dict[str, float]:
    """
    Compute pairwise Dice across frames (group Dice).
    
    Args:
        masks: (T, H, W) warped masks
        labels: List of label values to evaluate
    
    Returns:
        Dictionary with per-label and average Dice
    
    Note:
        This computes frame-to-frame Dice as a proxy for groupwise
        registration quality. See parent codebase for full metric details.
        
        Interpretation guidance:
        - Higher is better
        - This is a SANITY CHECK metric only
        - Not suitable for rigorous method comparison
    """
    T, H, W = masks.shape
    
    if T < 2:
        return {'dice_avg': float('nan')}
    
    results = {}
    all_dice = []
    
    for label in labels:
        label_dice = []
        for i in range(T):
            for j in range(i + 1, T):
                dice = compute_dice_simple(masks[i], masks[j], label=label)
                label_dice.append(dice)
        
        if label_dice:
            mean_dice = np.mean(label_dice)
            results[f'dice_label_{label}'] = mean_dice
            all_dice.extend(label_dice)
    
    if all_dice:
        results['dice_avg'] = np.mean(all_dice)
    else:
        results['dice_avg'] = float('nan')
    
    return results


def compute_displacement_stats(disp: torch.Tensor) -> Dict[str, float]:
    """
    Compute basic displacement field statistics.
    
    Args:
        disp: (T, 2, H, W) displacement field
    
    Returns:
        Dictionary with basic stats (mean magnitude, std, etc.)
    
    Note:
        These are descriptive statistics only, not quality metrics.
        They help verify that the displacement field is reasonable.
    """
    magnitude = torch.sqrt((disp ** 2).sum(dim=1))  # (T, H, W)
    
    stats = {
        'disp_magnitude_mean': magnitude.mean().item(),
        'disp_magnitude_std': magnitude.std().item(),
        'disp_magnitude_max': magnitude.max().item(),
        'disp_x_mean': disp[:, 0].mean().item(),
        'disp_y_mean': disp[:, 1].mean().item(),
    }
    
    return stats


def evaluate_case(
    result: Dict,
    masks_original: Optional[torch.Tensor] = None,
    method_name: str = "unknown",
) -> Dict[str, float]:
    """
    Evaluate a single case with basic sanity checks.
    
    Args:
        result: Dictionary from inference containing 'disp', 'warped', etc.
        masks_original: Optional (T, H, W) original masks for Dice computation
        method_name: Method identifier for labeling
    
    Returns:
        Dictionary of metrics with clear labeling
    
    Note:
        All metrics are labeled as SANITY CHECKS. The raw baseline is
        additionally marked as REFERENCE-ONLY to prevent misinterpretation
        as a competitive method.
    """
    metrics = {
        'method': method_name,
        'metric_type': 'SANITY_CHECK',
    }
    
    # Mark raw baseline explicitly
    if 'raw' in method_name.lower() or 'baseline' in method_name.lower():
        metrics['note'] = 'REFERENCE-ONLY (not a competitive method)'
    
    # Displacement statistics
    if 'disp' in result:
        disp_stats = compute_displacement_stats(result['disp'])
        metrics.update(disp_stats)
    
    # Dice metrics if masks available
    if masks_original is not None and 'disp' in result:
        # Need to warp masks (simplified here)
        # Full implementation would use proper mask warping with nearest interpolation
        warped_masks = masks_original  # Placeholder
        dice_metrics = compute_pairwise_dice(warped_masks)
        for k, v in dice_metrics.items():
            metrics[f'mask_{k}'] = v
    
    return metrics


def format_results_table(metrics_list: list) -> str:
    """
    Format list of metrics into simple text table.
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Formatted string table
    
    Note:
        This is a minimal formatting utility. For publication-quality
        tables, use dedicated tools like pandas or LaTeX generators.
    """
    if not metrics_list:
        return "No results to display"
    
    lines = ["=" * 60, "SANITY CHECK RESULTS (NOT FOR RIGOROUS COMPARISON)", "=" * 60, ""]
    
    for idx, metrics in enumerate(metrics_list):
        lines.append(f"Result {idx + 1}: {metrics.get('method', 'unknown')}")
        if 'note' in metrics:
            lines.append(f"  Note: {metrics['note']}")
        
        for key, val in metrics.items():
            if key in ('method', 'metric_type', 'note'):
                continue
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("IMPORTANT: These are sanity check metrics only.")
    lines.append("For rigorous evaluation, see the paper and full benchmark suite.")
    lines.append("=" * 60)
    
    return "\n".join(lines)
