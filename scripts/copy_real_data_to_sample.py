"""Copy real STONE/ACDC data into sample_data/. Run from repo root: python scripts/copy_real_data_to_sample.py"""
from pathlib import Path
import shutil
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_STONE = REPO_ROOT / "sample_data" / "stone"
SAMPLE_ACDC = REPO_ROOT / "sample_data" / "acdc"

# Real data roots (adjust if your paths differ)
REAL_STONE = Path("/data/Data/stone")
REAL_STONE_FEATURE = Path("/data/Projects/results/stone")
REAL_ACDC = Path("/data/Projects/cmr_reverse/results/acdc_eval_pack")

FEATURE_KEY = "logits_final"
CROP_HW = (192, 192)


def _center_crop_pad(feat: np.ndarray, th: int, tw: int) -> np.ndarray:
    *leading, h, w = feat.shape
    if h >= th and w >= tw:
        top, left = (h - th) // 2, (w - tw) // 2
        return feat[..., top : top + th, left : left + tw].copy()
    out = np.zeros((*leading, th, tw), dtype=feat.dtype)
    sh, sw = min(th, h), min(tw, w)
    top = (th - sh) // 2
    left = (tw - sw) // 2
    src_top = (h - th) // 2 if h >= th else 0
    src_left = (w - tw) // 2 if w >= tw else 0
    out[..., top : top + sh, left : left + sw] = feat[..., src_top : src_top + sh, src_left : src_left + sw]
    return out


def _copy_feature_reduced(src_npz: Path, dst_npz: Path) -> None:
    """Save only logits_final, center-cropped to 192x192 (smaller file)."""
    data = np.load(src_npz)
    if FEATURE_KEY not in data:
        raise KeyError(f"'{FEATURE_KEY}' not in {src_npz}")
    feat = _center_crop_pad(data[FEATURE_KEY], *CROP_HW)
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, **{FEATURE_KEY: feat})


def main():
    if not (REAL_STONE / "data").exists():
        print(f"[SKIP] STONE data not found: {REAL_STONE / 'data'}")
        return
    if not (REAL_ACDC / "data").exists():
        print(f"[SKIP] ACDC data not found: {REAL_ACDC / 'data'}")
        return

    (SAMPLE_STONE / "data").mkdir(parents=True, exist_ok=True)
    (SAMPLE_STONE / "features").mkdir(parents=True, exist_ok=True)
    (SAMPLE_ACDC / "data").mkdir(parents=True, exist_ok=True)
    if (REAL_ACDC / "feature").exists():
        (SAMPLE_ACDC / "feature").mkdir(parents=True, exist_ok=True)

    # STONE: one case 138_4
    stone_src = list((REAL_STONE / "data").glob("*138_4*.nii.gz"))
    if not stone_src:
        print("[SKIP] No STONE case matching 138_4 in", REAL_STONE / "data")
    else:
        dst = SAMPLE_STONE / "data" / stone_src[0].name
        shutil.copy2(stone_src[0], dst)
        print("Copied", stone_src[0].name, "->", dst)

    # STONE features for 138_4 (notebook expects base_name_features.npz, base_name = 0138_4 from 0138_4_0000.nii.gz)
    if REAL_STONE_FEATURE.exists():
        for name in ["0138_4_features.npz", "0138_4_0000_features.npz"]:
            f = REAL_STONE_FEATURE / name
            if f.exists():
                _copy_feature_reduced(f, SAMPLE_STONE / "features" / f.name)
                print("Copied (reduced)", f.name, "-> sample_data/stone/features/")
                break
        else:
            print("[SKIP] No 0138_4_features.npz in", REAL_STONE_FEATURE)

    # ACDC: first subject by name order, copy only the 4th slice (slice003, 1-based 4th)
    for old in (SAMPLE_ACDC / "data").glob("*.nii.gz"):
        old.unlink()
    for old in (SAMPLE_ACDC / "feature").glob("*.npz"):
        old.unlink()
    acdc_all = sorted((REAL_ACDC / "data").glob("*.nii.gz"))
    subject_files = []
    if not acdc_all:
        print("[SKIP] No ACDC data in", REAL_ACDC / "data")
    else:
        first_subject = acdc_all[0].name.split("_")[0]
        subject_files = sorted([f for f in acdc_all if f.name.startswith(first_subject)])
        # 4th slice = index 3 (slice000=1st, slice001=2nd, slice002=3rd, slice003=4th)
        slice_idx = 3
        if len(subject_files) > slice_idx:
            f = subject_files[slice_idx]
            shutil.copy2(f, SAMPLE_ACDC / "data" / f.name)
            print("Copied", f.name, "-> sample_data/acdc/data/ (4th slice)")
        else:
            print(f"[SKIP] First subject has only {len(subject_files)} slices, need 4th (index 3)")

    # ACDC feature for the 4th slice only (reduced: logits_final only, 192x192)
    slice_idx = 3
    if (REAL_ACDC / "feature").exists() and len(subject_files) > slice_idx:
        first_subject = acdc_all[0].name.split("_")[0]
        feat_files = sorted((REAL_ACDC / "feature").glob(f"{first_subject}_*_features.npz"))
        if len(feat_files) > slice_idx:
            f = feat_files[slice_idx]
            _copy_feature_reduced(f, SAMPLE_ACDC / "feature" / f.name)
            print("Copied (reduced)", f.name, "-> sample_data/acdc/feature/")

    print("Done. Run the notebook again to use real images in sample_data/.")


if __name__ == "__main__":
    main()
