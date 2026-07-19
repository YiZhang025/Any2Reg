"""Export nnUNet v1 features and convert ACDC frames to Any2Reg slice-time cases.

Input files must follow ``<patient>_frame<index>_0000.nii.gz``. The patched
nnUNet predictor writes one ``*_features.npz`` per frame. This script then
stacks every spatial slice over time and writes Any2Reg-compatible arrays in
``(C, T, H, W)`` layout.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


CASE_RE = re.compile(r"^(?P<patient>.+)_frame(?P<frame>\d+)$")


def parse_case_id(case_id: str) -> Tuple[str, int]:
    match = CASE_RE.match(case_id)
    if not match:
        raise ValueError(f"Unrecognized case id format: {case_id}")
    return match.group("patient"), int(match.group("frame"))


def run_prediction(
    input_folder: Path,
    pred_folder: Path,
    task_name: str,
    trainer: str,
    model: str,
    folds: List[int],
    overwrite: bool,
    save_npz: bool,
    save_features: bool,
    disable_tta: bool,
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
    extra_env: Dict[str, Optional[str]],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(input_folder),
        "-o",
        str(pred_folder),
        "-t",
        task_name,
        "-tr",
        trainer,
        "-m",
        model,
        "-f",
        *[str(f) for f in folds],
        "--num_threads_preprocessing",
        str(num_threads_preprocessing),
        "--num_threads_nifti_save",
        str(num_threads_nifti_save),
    ]
    if overwrite:
        cmd.append("--overwrite_existing")
    if save_npz:
        cmd.append("-z")
    if save_features:
        cmd.append("--save_features")
    if disable_tta:
        cmd.append("--disable_tta")

    env = os.environ.copy()
    env.update({key: value for key, value in extra_env.items() if value})
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def discover_cases(input_folder: Path) -> Dict[str, Dict[int, Path]]:
    grouped: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for p in sorted(input_folder.glob("*_0000.nii.gz")):
        case_id = p.name.replace("_0000.nii.gz", "")
        patient, frame_idx = parse_case_id(case_id)
        grouped[patient][frame_idx] = p
    if not grouped:
        raise RuntimeError(f"No *_0000.nii.gz found in {input_folder}")
    return grouped


def load_segmentation(pred_folder: Path, case_id: str) -> nib.Nifti1Image:
    seg_path = pred_folder / f"{case_id}.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation: {seg_path}")
    return nib.load(str(seg_path))


def load_segmentation_optional(pred_folder: Optional[Path], case_id: str) -> Optional[nib.Nifti1Image]:
    if pred_folder is None:
        return None
    seg_path = pred_folder / f"{case_id}.nii.gz"
    if not seg_path.exists():
        return None
    return nib.load(str(seg_path))


def load_softmax_npz(pred_folder: Path, case_id: str) -> Dict[str, np.ndarray]:
    npz_path = pred_folder / f"{case_id}.npz"
    if not npz_path.exists():
        return {}
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_feature_npz(pred_folder: Path, case_id: str) -> Dict[str, np.ndarray]:
    feat_path = pred_folder / f"{case_id}_features.npz"
    if not feat_path.exists():
        return {}
    with np.load(feat_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def stack_time_for_slice(
    arr_list: List[np.ndarray],
    slice_idx: int,
    z_axis: int,
) -> np.ndarray:
    slices = []
    for arr in arr_list:
        one_slice = np.take(arr, indices=slice_idx, axis=z_axis)
        slices.append(one_slice)
    return np.stack(slices, axis=-1)


def to_cthw(feature_slice_time: np.ndarray) -> np.ndarray:
    """
    Normalize stacked slice-time feature array to (C, T, H, W),
    which is what reg/corrmlp2d/inference_application/dataset.py expects.
    Input is typically:
    - (C, H, W, T) from CxZxHxW features sliced on Z and stacked on time
    - (H, W, T) from ZxHxW features
    """
    if feature_slice_time.ndim == 4:
        # (C, H, W, T) -> (C, T, H, W)
        return np.transpose(feature_slice_time, (0, 3, 1, 2))
    if feature_slice_time.ndim == 3:
        # (H, W, T) -> (1, T, H, W)
        return np.transpose(feature_slice_time, (2, 0, 1))[None, ...]
    raise ValueError(
        f"Unsupported stacked feature ndim={feature_slice_time.ndim}, shape={feature_slice_time.shape}"
    )


def pick_z_axis(arr: np.ndarray, z_size: int) -> int:
    # nnUNet feature/softmax exports normally use (C, Z, H, W).
    if arr.ndim == 4 and arr.shape[1] == z_size:
        return 1
    # A single-channel export may be stored as (Z, H, W).
    if arr.ndim == 3 and arr.shape[0] == z_size:
        return 0
    candidates = [i for i, s in enumerate(arr.shape) if s == z_size]
    if not candidates:
        raise ValueError(f"Cannot find z axis with size {z_size} in shape {arr.shape}")
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous z axis for z_size={z_size} in shape {arr.shape}; "
            "expected (C, Z, H, W) or (Z, H, W)"
        )
    return candidates[0]


def convert_frame_to_slice_time(
    input_folder: Path,
    pred_folder: Path,
    seg_orig_pred_folder: Optional[Path],
    out_root: Path,
    seg_dir_name: str,
    seg_orig_dir_name: str,
    write_softmax_as_data: bool,
    require_features: bool,
    max_patients: int,
    max_frames_per_patient: int,
) -> None:
    out_data = out_root / "data"
    out_seg = out_root / seg_dir_name
    out_seg_orig = out_root / seg_orig_dir_name
    out_feature = out_root / "feature"
    out_data.mkdir(parents=True, exist_ok=True)
    out_seg.mkdir(parents=True, exist_ok=True)
    if seg_orig_pred_folder is not None:
        out_seg_orig.mkdir(parents=True, exist_ok=True)
    out_feature.mkdir(parents=True, exist_ok=True)

    grouped = discover_cases(input_folder)
    patients = sorted(grouped.keys())
    if max_patients > 0:
        patients = patients[:max_patients]
    print(f"Found {len(patients)} patients to convert.")

    written_case_ids: List[str] = []
    for patient in patients:
        frame_dict = grouped[patient]
        frame_ids = sorted(frame_dict.keys())
        if max_frames_per_patient > 0:
            frame_ids = frame_ids[:max_frames_per_patient]
        if len(frame_ids) == 0:
            continue
        case_ids = [frame_dict[idx].name.replace("_0000.nii.gz", "") for idx in frame_ids]

        img_niftis = [nib.load(str(frame_dict[idx])) for idx in frame_ids]
        img_arrays = [img.get_fdata(dtype=np.float32) for img in img_niftis]
        seg_niftis = [load_segmentation(pred_folder, cid) for cid in case_ids]
        seg_arrays = [seg.get_fdata(dtype=np.float32).astype(np.int16) for seg in seg_niftis]
        seg_orig_niftis = [load_segmentation_optional(seg_orig_pred_folder, cid) for cid in case_ids]
        has_seg_orig = all(x is not None for x in seg_orig_niftis)
        if seg_orig_pred_folder is not None and not has_seg_orig:
            missing = [case_ids[i] for i, x in enumerate(seg_orig_niftis) if x is None]
            show = ", ".join(missing[:5])
            if len(missing) > 5:
                show += ", ..."
            raise RuntimeError(
                f"Missing seg_orig nifti in {seg_orig_pred_folder} for patient {patient}. "
                f"Missing {len(missing)} frame(s): {show}"
            )
        seg_orig_arrays = None
        if has_seg_orig:
            seg_orig_arrays = [
                x.get_fdata(dtype=np.float32).astype(np.int16)  # type: ignore[union-attr]
                for x in seg_orig_niftis
            ]

        shape0 = img_arrays[0].shape
        if len(shape0) != 3:
            raise ValueError(f"Expected 3D frame volume, got shape {shape0} for {case_ids[0]}")
        if any(a.shape != shape0 for a in img_arrays):
            raise ValueError(f"Inconsistent image shapes in patient {patient}")
        if any(a.shape != shape0 for a in seg_arrays):
            raise ValueError(f"Inconsistent segmentation shapes in patient {patient}")
        if seg_orig_arrays is not None and any(a.shape != shape0 for a in seg_orig_arrays):
            raise ValueError(f"Inconsistent seg_orig shapes in patient {patient}")

        z_size = shape0[2]
        affine = img_niftis[0].affine

        softmax_per_frame = [load_softmax_npz(pred_folder, cid) for cid in case_ids]
        feature_per_frame = [load_feature_npz(pred_folder, cid) for cid in case_ids]

        has_softmax = all(len(d) > 0 for d in softmax_per_frame)
        has_features = all(len(d) > 0 for d in feature_per_frame)
        if require_features and not has_features:
            missing = [
                case_ids[i]
                for i, d in enumerate(feature_per_frame)
                if len(d) == 0
            ]
            show = ", ".join(missing[:5])
            if len(missing) > 5:
                show += ", ..."
            raise RuntimeError(
                f"Missing feature npz in {pred_folder} for patient {patient}. "
                f"Missing {len(missing)} frame(s): {show}. "
                "Please run nnUNet_predict with --save_features."
            )

        for z in range(z_size):
            slice_id = f"{patient}_slice{z:03d}"

            img_t = stack_time_for_slice(img_arrays, z, z_axis=2)
            seg_t = stack_time_for_slice(seg_arrays, z, z_axis=2)

            nib.save(nib.Nifti1Image(img_t, affine), str(out_data / f"{slice_id}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(seg_t.astype(np.int16), affine), str(out_seg / f"{slice_id}.nii.gz"))
            if seg_orig_arrays is not None:
                seg_orig_t = stack_time_for_slice(seg_orig_arrays, z, z_axis=2)
                nib.save(
                    nib.Nifti1Image(seg_orig_t.astype(np.int16), affine),
                    str(out_seg_orig / f"{slice_id}.nii.gz"),
                )
            written_case_ids.append(f"{slice_id}_0000.nii.gz")

            if has_softmax and write_softmax_as_data:
                softmax_slice_pack = {}
                for key in softmax_per_frame[0]:
                    arr0 = softmax_per_frame[0][key]
                    z_axis = pick_z_axis(arr0, z_size)
                    arr_list = [d[key] for d in softmax_per_frame]
                    stacked = stack_time_for_slice(arr_list, z, z_axis=z_axis)
                    softmax_slice_pack[key] = to_cthw(stacked)
                np.savez_compressed(out_data / f"{slice_id}_softmax.npz", **softmax_slice_pack)

            if has_features:
                feature_slice_pack = {}
                for key in feature_per_frame[0]:
                    arr0 = feature_per_frame[0][key]
                    z_axis = pick_z_axis(arr0, z_size)
                    arr_list = [d[key] for d in feature_per_frame]
                    stacked = stack_time_for_slice(arr_list, z, z_axis=z_axis)
                    feature_slice_pack[key] = to_cthw(stacked)
                np.savez_compressed(out_feature / f"{slice_id}_features.npz", **feature_slice_pack)

        print(
            f"{patient}: frames={len(frame_ids)}, slices={z_size}, "
            f"softmax={'yes' if has_softmax else 'no'}, "
            f"features={'yes' if has_features else 'no'}, "
            f"seg_orig={'yes' if seg_orig_arrays is not None else 'no'}"
        )

    written_case_ids = sorted(set(written_case_ids))
    case_id_file = out_root / "case_ids_all.txt"
    with open(case_id_file, "w", encoding="utf-8") as f:
        for cid in written_case_ids:
            f.write(f"{cid}\n")

    split_json = {
        "train": [],
        "val": [],
        "test": written_case_ids,
    }
    with open(out_root / "acdc_splits.json", "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2)

    print(f"Wrote case list: {case_id_file} ({len(written_case_ids)} cases)")
    print(f"Wrote split json: {out_root / 'acdc_splits.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ACDC frame inference then convert to slice-wise HxWxT stacks."
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Path to Task027 imagesTs with *_0000.nii.gz files.",
    )
    parser.add_argument(
        "--pred-folder",
        type=Path,
        required=True,
        help="Per-frame RI prediction output directory (used for seg + feature export).",
    )
    parser.add_argument(
        "--seg-orig-pred-folder",
        type=Path,
        default=None,
        help="Optional per-frame official nnUNet prediction directory (used for seg_orig export).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Converted output root containing data/feature/seg/seg_orig.",
    )
    parser.add_argument("--seg-dir-name", default="seg", help="Output folder name for RI segmentation.")
    parser.add_argument("--seg-orig-dir-name", default="seg_orig", help="Output folder name for official segmentation.")
    parser.add_argument("--task-name", default="Task900_ACDC_Phys")
    parser.add_argument("--trainer", default="nnUNetTrainerV2_InvGreAug")
    parser.add_argument("--model", default="2d")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Pass --disable_tta to nnUNet prediction (used for the released ACDC features).",
    )
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--save-features",
        dest="save_features",
        action="store_true",
        help="Export *_features.npz during prediction (default).",
    )
    feature_group.add_argument(
        "--no-save-features",
        dest="save_features",
        action="store_false",
        help="Disable feature export; mainly useful with --allow-missing-features.",
    )
    parser.set_defaults(save_features=True)
    parser.add_argument("--num-threads-preprocessing", type=int, default=6)
    parser.add_argument("--num-threads-nifti-save", type=int, default=2)
    parser.add_argument(
        "--nnunet-raw-data-base",
        default=os.environ.get("nnUNet_raw_data_base"),
        help="nnUNet raw-data root; defaults to $nnUNet_raw_data_base.",
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        default=os.environ.get("nnUNet_preprocessed"),
        help="nnUNet preprocessed root; defaults to $nnUNet_preprocessed.",
    )
    parser.add_argument(
        "--results-folder",
        default=os.environ.get("RESULTS_FOLDER"),
        help="nnUNet model-results root; defaults to $RESULTS_FOLDER.",
    )
    parser.add_argument(
        "--write-softmax-as-data",
        action="store_true",
        help="If enabled, write per-slice softmax npz into data/ as *_softmax.npz.",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="If set, do not fail when *_features.npz is missing in pred folder.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="Debug only. >0 means convert only first N patients.",
    )
    parser.add_argument(
        "--max-frames-per-patient",
        type=int,
        default=0,
        help="Debug only. >0 means convert only first N frames for each patient.",
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input_folder}")

    env = {
        "nnUNet_raw_data_base": args.nnunet_raw_data_base,
        "nnUNet_preprocessed": args.nnunet_preprocessed,
        "RESULTS_FOLDER": args.results_folder,
    }
    if not args.skip_predict:
        missing_env = [key for key, value in env.items() if not value]
        if missing_env:
            joined = ", ".join(missing_env)
            raise SystemExit(
                f"Missing nnUNet path setting(s): {joined}. "
                "Set the environment variables or pass the matching CLI options."
            )
        args.pred_folder.mkdir(parents=True, exist_ok=True)
    elif not args.pred_folder.is_dir():
        raise SystemExit(f"Prediction folder does not exist: {args.pred_folder}")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    if not args.skip_predict:
        run_prediction(
            input_folder=args.input_folder,
            pred_folder=args.pred_folder,
            task_name=args.task_name,
            trainer=args.trainer,
            model=args.model,
            folds=args.folds,
            overwrite=args.overwrite,
            save_npz=args.save_npz,
            save_features=args.save_features,
            disable_tta=args.disable_tta,
            num_threads_preprocessing=args.num_threads_preprocessing,
            num_threads_nifti_save=args.num_threads_nifti_save,
            extra_env=env,
        )

    convert_frame_to_slice_time(
        input_folder=args.input_folder,
        pred_folder=args.pred_folder,
        seg_orig_pred_folder=args.seg_orig_pred_folder,
        out_root=args.output_folder,
        seg_dir_name=args.seg_dir_name,
        seg_orig_dir_name=args.seg_orig_dir_name,
        write_softmax_as_data=args.write_softmax_as_data,
        require_features=not args.allow_missing_features,
        max_patients=args.max_patients,
        max_frames_per_patient=args.max_frames_per_patient,
    )


if __name__ == "__main__":
    main()
