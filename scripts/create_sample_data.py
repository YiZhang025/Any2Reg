"""Generate synthetic sample data. Run from repo root: python scripts/create_sample_data.py"""
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise SystemExit("nibabel required: pip install nibabel")

REPO_ROOT = Path(__file__).resolve().parents[1]
STONE_DATA = REPO_ROOT / "sample_data" / "stone" / "data"
STONE_FEAT = REPO_ROOT / "sample_data" / "stone" / "features"
ACDC_DATA = REPO_ROOT / "sample_data" / "acdc" / "data"

def main():
    STONE_DATA.mkdir(parents=True, exist_ok=True)
    STONE_FEAT.mkdir(parents=True, exist_ok=True)
    ACDC_DATA.mkdir(parents=True, exist_ok=True)

    # STONE: one volume 192x192x11 (H, W, T), name contains 138_4
    h, w, t = 192, 192, 11
    stone_vol = np.random.rand(h, w, t).astype(np.float32) * 0.5 + 0.2
    stone_path = STONE_DATA / "0138_4_0000.nii.gz"
    nib.save(nib.Nifti1Image(stone_vol, np.eye(4)), stone_path)
    print(f"Created {stone_path}")

    # STONE features: (1, T, H, W) for logits_final
    stone_feat = np.random.randn(1, t, h, w).astype(np.float32) * 0.1
    stone_feat_path = STONE_FEAT / "0138_4_0000_features.npz"
    np.savez_compressed(stone_feat_path, logits_final=stone_feat)
    print(f"Created {stone_feat_path}")

    # ACDC: first subject, 3 slices so mid-slice is index 1
    for i, slice_id in enumerate(["slice001", "slice002", "slice003"]):
        vol = np.random.rand(192, 192, 30).astype(np.float32) * 0.5 + 0.2
        path = ACDC_DATA / f"patient001_{slice_id}_0000.nii.gz"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), path)
        print(f"Created {path}")

    print("Sample data ready under sample_data/")

if __name__ == "__main__":
    main()
