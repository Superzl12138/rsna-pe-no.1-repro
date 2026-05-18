"""
Build an RSNA-style external labels CSV from class-organized hospital DICOM data.

Expected root layout example:
    /data/zilizhu/PE/CJ-DATA/
        APE/<case>/<study_dir>/<series_dir>/*.dcm
        CPE/<case>/<study_dir>/<series_dir>/*.dcm
        D_慢加急/<case>/<study_dir>/<series_dir>/*.dcm
        ctpa-阴性/<case>/<study_dir>/<series_dir>/*.dcm

The script scans leaf directories under the class folders, reads the first valid
DICOM in each directory, extracts Study/Series UID, and writes one CSV row per
series. This is for exam-level external validation, not image-level labels.

Example:
    python build_external_labels_csv.py \
        --root-dir /data/zilizhu/PE/CJ-DATA \
        --output-csv ./external_labels.csv
"""

import argparse
import csv
import os
from pathlib import Path

import pydicom


CLASS_TO_LABELS = {
    "APE": {
        "external_case_label": "acute",
        "negative_exam_for_pe": 0,
        "chronic_pe": 0,
        "acute_and_chronic_pe": 0,
    },
    "CPE": {
        "external_case_label": "chronic",
        "negative_exam_for_pe": 0,
        "chronic_pe": 1,
        "acute_and_chronic_pe": 0,
    },
    "D_慢加急": {
        "external_case_label": "acute_and_chronic",
        "negative_exam_for_pe": 0,
        "chronic_pe": 0,
        "acute_and_chronic_pe": 1,
    },
    "NEG": {
        "external_case_label": "negative",
        "negative_exam_for_pe": 1,
        "chronic_pe": 0,
        "acute_and_chronic_pe": 0,
    },
    "阴性": {
        "external_case_label": "negative",
        "negative_exam_for_pe": 1,
        "chronic_pe": 0,
        "acute_and_chronic_pe": 0,
    },
    "ctpa-阴性": {
        "external_case_label": "negative",
        "negative_exam_for_pe": 1,
        "chronic_pe": 0,
        "acute_and_chronic_pe": 0,
    },
}


def is_hidden(name):
    return name.startswith(".")


def try_read_dicom(file_path):
    try:
        dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        if not hasattr(dataset, "StudyInstanceUID") or not hasattr(dataset, "SeriesInstanceUID"):
            return None
        return dataset
    except Exception:
        return None


def iter_leaf_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [name for name in dirnames if not is_hidden(name)]
        visible_files = [
            name for name in filenames
            if not is_hidden(name) and os.path.isfile(os.path.join(dirpath, name))
        ]
        if not visible_files:
            continue
        if dirnames:
            continue
        yield Path(dirpath)


def get_first_valid_dataset(series_dir):
    file_paths = sorted(
        file_path for file_path in series_dir.iterdir()
        if file_path.is_file() and not is_hidden(file_path.name)
    )
    for file_path in file_paths:
        dataset = try_read_dicom(file_path)
        if dataset is not None:
            return file_path, dataset, len(file_paths)
    return None, None, len(file_paths)


def build_record(root_dir, series_dir):
    rel_path = series_dir.relative_to(root_dir)
    parts = rel_path.parts
    if len(parts) < 1:
        return None

    class_name = parts[0]
    if class_name not in CLASS_TO_LABELS:
        return None

    first_valid_file, dataset, num_files = get_first_valid_dataset(series_dir)
    if dataset is None:
        raise RuntimeError(f"No valid DICOM with Study/Series UID found in {series_dir}")

    case_dirname = parts[1] if len(parts) >= 2 else ""
    study_dirname = parts[-2] if len(parts) >= 2 else ""
    series_dirname = parts[-1]
    label_info = CLASS_TO_LABELS[class_name]

    return {
        "StudyInstanceUID": str(getattr(dataset, "StudyInstanceUID")),
        "SeriesInstanceUID": str(getattr(dataset, "SeriesInstanceUID")),
        "class_name": class_name,
        "external_case_label": label_info["external_case_label"],
        "negative_exam_for_pe": label_info["negative_exam_for_pe"],
        "chronic_pe": label_info["chronic_pe"],
        "acute_and_chronic_pe": label_info["acute_and_chronic_pe"],
        "case_dirname": case_dirname,
        "study_dirname": study_dirname,
        "series_dirname": series_dirname,
        "num_dicoms": num_files,
        "source_path": str(series_dir),
        "example_dicom_path": str(first_valid_file),
    }


def write_csv(output_csv, records):
    fieldnames = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "class_name",
        "external_case_label",
        "negative_exam_for_pe",
        "chronic_pe",
        "acute_and_chronic_pe",
        "case_dirname",
        "study_dirname",
        "series_dirname",
        "num_dicoms",
        "source_path",
        "example_dicom_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Build an external labels CSV from class-organized DICOM directories.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing class folders such as APE/CPE/D_慢加急/ctpa-阴性.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    records = []
    skipped_leaf_dirs = []
    for series_dir in sorted(iter_leaf_dirs(root_dir)):
        record = build_record(root_dir, series_dir)
        if record is None:
            skipped_leaf_dirs.append(str(series_dir))
            continue
        records.append(record)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, records)

    unique_pairs = {(row["StudyInstanceUID"], row["SeriesInstanceUID"]) for row in records}
    print(f"Scanned labeled series: {len(records)}")
    print(f"Unique Study/Series pairs: {len(unique_pairs)}")
    print(f"Skipped leaf dirs outside known classes: {len(skipped_leaf_dirs)}")
    print(f"Output CSV: {output_csv}")
    for class_name in sorted(CLASS_TO_LABELS):
        count = sum(1 for row in records if row["class_name"] == class_name)
        if count > 0:
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
