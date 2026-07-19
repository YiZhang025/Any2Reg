# Feature generation

Any2Reg's `raw+logits` checkpoint expects a precomputed feature file next to
each input case. The feature archive must contain `logits_final` in
`(C, T, H, W)` order. During demo loading, the channels are averaged to the
single feature channel used by the released checkpoint.

The feature pipeline used for the released experiments has two parts:

1. a small nnUNet v1 patch that exports pre-softmax logits during inference;
2. an ACDC converter that stacks per-frame 3D predictions into slice-wise
   time series.

The segmentation model weights and medical datasets are not bundled in this
repository. The pretrained `Task900_ACDC_Phys` weights are publicly available
through the Reverse Imaging repository, as described below.

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

Download the public Reverse Imaging nnUNet weights:

```bash
git clone --recurse-submodules https://github.com/Ido-zh/cmr_reverse.git
cd cmr_reverse
bash download_nnunet_weights.sh
```

The downloader uses `$RESULTS_FOLDER` and installs `Task900_ACDC_Phys` under
`$RESULTS_FOLDER/nnUNet/2d/`. See the upstream
[`cmr_reverse` instructions](https://github.com/Ido-zh/cmr_reverse#1-download-weights-and-required-data)
and
[`download_nnunet_weights.sh`](https://github.com/Ido-zh/cmr_reverse/blob/main/download_nnunet_weights.sh)
for the maintained download location.

## 2. Export features for group-volume inputs

For inputs already stored as one `*_0000.nii.gz` group volume, run the patched
predictor directly:

```bash
python -m nnunet.inference.predict_simple \
  -i /path/to/input \
  -o /path/to/predictions \
  -t Task900_ACDC_Phys \
  -tr nnUNetTrainerV2_InvGreAug \
  -m 2d \
  -f 0 1 2 3 4 \
  --save_features \
  --disable_tta
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
  --task-name Task900_ACDC_Phys \
  --trainer nnUNetTrainerV2_InvGreAug \
  --model 2d \
  --disable-tta
```

`Task900_ACDC_Phys`, `nnUNetTrainerV2_InvGreAug`, five folds, and disabled TTA
are the settings used to generate the released ACDC feature input. The task and
trainer are also the script defaults; `--disable-tta` remains explicit so that
switching TTA on is an intentional choice. Download the model weights using
the Reverse Imaging command above.

The script exports features by default. Pass the three nnUNet paths as CLI
options instead of environment variables if preferred. If frame predictions
and `*_features.npz` files already exist, add `--skip-predict`.

Feed complete per-frame volumes to nnUNet, then let this script split them into
slice-time cases. Cropping a volume to one slice before prediction changes the
nnUNet intensity-normalization statistics and therefore changes its features.

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

See [`VALIDATION.md`](VALIDATION.md) for a real-data end-to-end check of this
pipeline and the released Any2Reg checkpoint.
