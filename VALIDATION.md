# Real-data pipeline validation

The released feature-generation path and Any2Reg checkpoint were revalidated
end to end on 2026-07-19 using an NVIDIA RTX 5090.

## Protocol

- Input: ACDC `patient101`, all 30 original cardiac frames and all 10 slices.
- Feature model: `Task900_ACDC_Phys`,
  `nnUNetTrainerV2_InvGreAug`, 2D, folds 0-4, `--disable-tta`.
- Feature export: the nnUNet v1 patch in `scripts/`, followed by
  `scripts/generate_acdc_features.py`.
- Registration: raw zero-displacement reference versus the released
  `raw+logits` Any2Reg checkpoint.
- Checkpoint SHA-256:
  `9a5fa715c0b32f3aea80e260ec5a1e89f0031ad4358a58f068a77361629e0b7c`.
- Dice labels: 1, 2, and 3, evaluated groupwise across the 30 frames.

The newly generated `patient101_slice003` inputs were also compared with the
historical experiment pack. The raw image, reference segmentation, and
`enc_stage0` were exactly equal. The maximum absolute differences were
`9.77e-4` for `dec_stage5` and `7.81e-3` for `logits_final`; the mean absolute
logit difference was `1.53e-4`.

## Results

| Method | Slice-average group Dice | 3D-volume group Dice |
| --- | ---: | ---: |
| Raw (zero displacement) | 0.766944 | 0.831504 |
| Any2Reg raw+logits | 0.834762 | 0.915540 |
| Absolute improvement | +0.067818 | +0.084036 |

These numbers are a reproducibility check on one subject, not a replacement
for the paper's complete evaluation. ACDC manual annotations are available
only at ED/ES. To measure dense 30-frame groupwise consistency, this check uses
the original experiment's per-frame nnUNet pseudo-labels (`seg_orig`), not
manual ground truth for every frame.
