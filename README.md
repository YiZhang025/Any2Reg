# Any2Reg — Supplementary Code

This repository provides minimal inference code for the groupwise registration method described in the paper with pretrained weights. It runs on one STONE and one ACDC case and exports displacement fields, metrics, and visualisations (PNG + GIF). Raw image is included for reference only.

**STONE (138_4)** — comparison of registration results (Raw, Elastix, GroupRegNet, MultiMorph, MultiMorph-Seg, Any²Reg w/o FM, Any²Reg, Any²Reg IO):

![STONE 138_4](figures/stone_138_4_combined_no_mask.gif)

---

**Requirements.** Python 3.8+, PyTorch 2.0+, and dependencies in `requirements.txt`. GPU optional.

**Setup.** Install with `pip install -r requirements.txt`. Place pretrained weights in `checkpoints/` (see `checkpoints/README.md`). The notebook uses `sample_data/` when present; otherwise set data paths in the first cell.

**Run.** Execute `notebooks/run_submission_demo.ipynb` from the repository root (or from `notebooks/` with parent on `sys.path`). Outputs are written to `outputs/run_YYYYMMDD_HHMMSS/`.

**Data.** STONE: NIfTI volumes in `data/`, optional precomputed features in `features/*_features.npz` (key `logits_final`). ACDC: same layout under `acdc/data` and `acdc/feature`. The demo expects one STONE subject (e.g. 138_4) and one ACDC slice; see `sample_data/README.md` for generating synthetic data.

**License.** MIT. Research use only.
