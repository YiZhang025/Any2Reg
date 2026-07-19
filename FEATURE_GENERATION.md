# Feature generation

Any2Reg's `raw+logits` checkpoint expects a precomputed feature file next to
each input case. The feature archive must contain `logits_final` in
`(C, T, H, W)` order. During demo loading, the channels are averaged to the
single feature channel used by the released checkpoint.

The feature pipeline used for the released experiments has two parts:

1. a small nnUNet v1 patch that exports pre-softmax logits during inference;
2. an ACDC converter that stacks per-frame 3D predictions into slice-wise
   time series.

The segmentation model weights and medical datasets are not distributed in
this repository.

## 1. Install the feature-exporting nnUNet v1 fork

The included patch is pinned to commit `ba0a16f` of
[`Ido-zh/nnUNet-phys-seg`](https://github.com/Ido-zh/nnUNet-phys-seg):

```bash
git clone https://github.com/Ido-zh/nnUNet-phys-seg.git
cd nnUNet-phys-seg
git checkout ba0a16f
git apply /path/to/Any2Reg/scripts/nnunet_v1_feature_export.patch
pip install -e .
```

The nnUNet-derived patch is distributed under Apache-2.0; see
[`scripts/NNUNET_PATCH_LICENSE`](scripts/NNUNET_PATCH_LICENSE).

The patch adds `--save_features` to `nnunet.inference.predict_simple` and
writes `<case>_features.npz`. It supports the 2D/pseudo-3D inference path used
in these experiments.

Configure the standard nnUNet v1 paths before prediction:

```bash
export nnUNet_raw_data_base=/path/to/nnunet/raw
export nnUNet_preprocessed=/path/to/nnunet/preprocessed
export RESULTS_FOLDER=/path/to/nnunet/results
```

## 2. Export features for group-volume inputs

For inputs already stored as one `*_0000.nii.gz` group volume, run the patched
predictor directly:

```bash
python -m nnunet.inference.predict_simple \
  -i /path/to/input \
  -o /path/to/predictions \
  -t Task027_ACDC \
  -tr nnUNetTrainerV2 \
  -m 2d \
  -f 0 1 2 3 4 \
  --save_features
```

Copy or link each generated `<case>_features.npz` into the feature directory
configured in the demo notebook.

## 3. Export and convert ACDC per-frame volumes

ACDC frames should use this naming convention:

```text
<patient>_frame<index>_0000.nii.gz
```

Run prediction and conversion together:

```bash
python scripts/generate_acdc_features.py \
  --input-folder /path/to/acdc/imagesTs \
  --pred-folder /path/to/frame_predictions \
  --output-folder /path/to/any2reg_acdc \
  --task-name Task027_ACDC \
  --trainer nnUNetTrainerV2 \
  --model 2d
```

The script exports features by default. You can pass the three nnUNet paths as
CLI options instead of environment variables. If frame predictions and
`*_features.npz` files already exist, add `--skip-predict`.

The converted directory contains:

```text
any2reg_acdc/
├── data/       # slice-wise H x W x T NIfTI inputs
├── feature/    # logits_final arrays in C x T x H x W order
├── seg/        # slice-wise predicted segmentations
├── case_ids_all.txt
└── acdc_splits.json
```

Feature archives can be large. `scripts/shrink_features.py` keeps only
`logits_final`, crops/pads it to the requested spatial size, and writes a
compressed copy.
