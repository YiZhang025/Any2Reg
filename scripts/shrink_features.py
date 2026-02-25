"""Reduce feature .npz size: keep only logits_final and center-crop to 192x192. Run from repo root."""
import argparse
import numpy as np
from pathlib import Path

FEATURE_KEY = "logits_final"
CROP_SIZE = (192, 192)


def _center_crop_pad(feat: np.ndarray, target_hw: tuple) -> np.ndarray:
    """(..., H, W) -> (..., target_h, target_w)."""
    *leading, h, w = feat.shape
    th, tw = target_hw
    if h >= th and w >= tw:
        top = (h - th) // 2
        left = (w - tw) // 2
        return feat[..., top : top + th, left : left + tw].copy()
    # Pad
    out = np.zeros((*leading, th, tw), dtype=feat.dtype)
    sh = min(th, h)
    sw = min(tw, w)
    top = (th - sh) // 2
    left = (tw - sw) // 2
    src_top = max(0, (h - th) // 2) if h < th else (h - th) // 2
    src_left = max(0, (w - tw) // 2) if w < tw else (w - tw) // 2
    out[..., top : top + sh, left : left + sw] = feat[..., src_top : src_top + sh, src_left : src_left + sw]
    return out


def shrink_npz(in_path: Path, out_path: Path, crop_hw: tuple = CROP_SIZE) -> None:
    data = np.load(in_path)
    if FEATURE_KEY not in data:
        raise KeyError(f"'{FEATURE_KEY}' not in {in_path}")
    feat = data[FEATURE_KEY]
    feat = _center_crop_pad(feat, crop_hw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{FEATURE_KEY: feat})
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path.name} ({feat.shape}, {size_mb:.2f} MB)")


def main():
    p = argparse.ArgumentParser(description="Shrink feature npz: keep logits_final only, crop to 192x192")
    p.add_argument("input", type=Path, help="Input .npz path")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output .npz path (default: input with _reduced suffix)")
    p.add_argument("--crop", type=int, default=192, help="Spatial crop size (default 192)")
    args = p.parse_args()
    in_path = args.input.resolve()
    if not in_path.is_file():
        raise SystemExit(f"Not a file: {in_path}")
    out_path = args.output
    if out_path is None:
        out_path = in_path.parent / (in_path.stem + "_reduced.npz")
    out_path = out_path.resolve()
    shrink_npz(in_path, out_path, crop_hw=(args.crop, args.crop))


if __name__ == "__main__":
    main()
