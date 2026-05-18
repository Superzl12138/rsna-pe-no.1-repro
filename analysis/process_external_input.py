"""
Build external-validation pickle files that mirror the RSNA process_input outputs,
but do not require image-level labels or the full RSNA train.csv schema.

Input:
    - A labels CSV with at least:
        StudyInstanceUID, SeriesInstanceUID
    - A standardized DICOM root in the layout:
        <train-root>/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

Output:
    - series_dict.pickle
    - image_dict.pickle
    - series_list_external.pickle
    - image_list_external.pickle

Example:
    python process_external_input.py \
        --labels-csv ./external_labels.csv \
        --train-root /data/zilizhu/PE/CJ-EXTERNAL-STANDARD/train \
        --output-dir ./external_pickles
"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


SERIES_LABEL_FIELDS = [
    "negative_exam_for_pe",
    "qa_motion",
    "qa_contrast",
    "flow_artifact",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "leftsided_pe",
    "chronic_pe",
    "true_filling_defect_not_pe",
    "rightsided_pe",
    "acute_and_chronic_pe",
    "central_pe",
    "indeterminate",
]


def safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip() == "":
            return float(default)
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def get_dicom_array(series_dir):
    dicom_files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    dicoms = [pydicom.dcmread(path, stop_before_pixels=True, force=True) for path in dicom_files]
    z_pos = []
    exposure = []
    thickness = []
    image_ids = []
    for dicom_path, dataset in zip(dicom_files, dicoms):
        image_ids.append(str(getattr(dataset, "SOPInstanceUID", Path(dicom_path).stem)))
        if hasattr(dataset, "ImagePositionPatient") and len(dataset.ImagePositionPatient) >= 3:
            z_pos.append(safe_float(dataset.ImagePositionPatient[-1], default=0.0))
        else:
            z_pos.append(0.0)
        exposure.append(safe_float(getattr(dataset, "Exposure", 0.0), default=0.0))
        thickness.append(safe_float(getattr(dataset, "SliceThickness", 0.0), default=0.0))

    sorted_idx = np.argsort(np.asarray(z_pos, dtype=np.float32), kind="mergesort")
    return (
        np.asarray(dicom_files, dtype=object)[sorted_idx],
        np.asarray(image_ids, dtype=object)[sorted_idx],
        np.asarray(z_pos, dtype=np.float32)[sorted_idx],
        np.asarray(exposure, dtype=np.float32)[sorted_idx],
        np.asarray(thickness, dtype=np.float32)[sorted_idx],
    )


def build_series_entry(label_row):
    series_entry = {}
    for field in SERIES_LABEL_FIELDS:
        series_entry[field] = safe_float(label_row.get(field, 0.0), default=0.0)
    series_entry["sorted_image_list"] = []
    series_entry["class_name"] = str(label_row.get("class_name", ""))
    series_entry["external_case_label"] = str(label_row.get("external_case_label", ""))
    series_entry["source_path"] = str(label_row.get("source_path", ""))
    return series_entry


def main():
    parser = argparse.ArgumentParser(description="Build external process_input-style pickles from standardized external DICOM directories.")
    parser.add_argument("--labels-csv", required=True, help="Path to external_labels.csv or external_labels_dedup.csv.")
    parser.add_argument("--train-root", required=True, help="Standardized train root, e.g. /data/.../CJ-EXTERNAL-STANDARD/train")
    parser.add_argument("--output-dir", required=True, help="Output directory for pickle files.")
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv).expanduser().resolve()
    train_root = Path(args.train_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_csv}")
    if not train_root.exists():
        raise FileNotFoundError(f"Train root does not exist: {train_root}")

    df = pd.read_csv(labels_csv)
    required_columns = {"StudyInstanceUID", "SeriesInstanceUID"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in labels CSV: {missing_columns}")

    df = df.drop_duplicates(subset=["StudyInstanceUID", "SeriesInstanceUID"], keep="first").copy()
    df = df.reset_index(drop=True)

    series_dict = {}
    image_dict = {}
    series_list_external = []

    for _, row in df.iterrows():
        study_uid = str(row["StudyInstanceUID"])
        series_uid = str(row["SeriesInstanceUID"])
        series_id = f"{study_uid}_{series_uid}"
        series_dict[series_id] = build_series_entry(row)
        series_list_external.append(series_id)

    series_list_external = sorted(series_list_external)
    print(f"Series rows loaded: {len(series_list_external)}")

    missing_series_dirs = []
    image_list_external = []

    for series_id in series_list_external:
        study_uid, series_uid = series_id.split("_", 1)
        series_dir = train_root / study_uid / series_uid
        if not series_dir.exists():
            missing_series_dirs.append(str(series_dir))
            continue

        file_list, image_ids, z_pos_list, exposure_list, thickness_list = get_dicom_array(str(series_dir))
        if len(file_list) == 0:
            missing_series_dirs.append(str(series_dir))
            continue

        sorted_image_list = [str(image_id) for image_id in image_ids]
        for i, image_id in enumerate(sorted_image_list):
            if i == 0:
                image_minus1 = image_id
                image_plus1 = sorted_image_list[i + 1] if len(sorted_image_list) > 1 else image_id
            elif i == len(sorted_image_list) - 1:
                image_minus1 = sorted_image_list[i - 1]
                image_plus1 = image_id
            else:
                image_minus1 = sorted_image_list[i - 1]
                image_plus1 = sorted_image_list[i + 1]

            image_dict[image_id] = {
                "pe_present_on_image": 0.0,
                "series_id": series_id,
                "z_pos": float(z_pos_list[i]),
                "exposure": float(exposure_list[i]),
                "thickness": float(thickness_list[i]),
                "image_minus1": image_minus1,
                "image_plus1": image_plus1,
            }
            image_list_external.append(image_id)

        series_dict[series_id]["sorted_image_list"] = sorted_image_list

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "series_dict.pickle", "wb") as handle:
        pickle.dump(series_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "image_dict.pickle", "wb") as handle:
        pickle.dump(image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "series_list_external.pickle", "wb") as handle:
        pickle.dump(series_list_external, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "image_list_external.pickle", "wb") as handle:
        pickle.dump(image_list_external, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Output dir: {output_dir}")
    print(f"series_dict size: {len(series_dict)}")
    print(f"image_dict size: {len(image_dict)}")
    print(f"series_list_external size: {len(series_list_external)}")
    print(f"image_list_external size: {len(image_list_external)}")
    print(f"Missing/empty series dirs: {len(missing_series_dirs)}")
    if missing_series_dirs:
        print("First 10 missing/empty series dirs:")
        for path in missing_series_dirs[:10]:
            print(f"  {path}")

    if series_list_external:
        sample_series_id = series_list_external[0]
        sample_image_list = series_dict[sample_series_id]["sorted_image_list"]
        print(f"Sample series_id: {sample_series_id}")
        print(series_dict[sample_series_id])
        if sample_image_list:
            print(f"Sample image_id: {sample_image_list[0]}")
            print(image_dict[sample_image_list[0]])


if __name__ == "__main__":
    main()
