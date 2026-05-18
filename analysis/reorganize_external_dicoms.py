"""
Reorganize external hospital DICOM files into the RSNA-style layout expected by
the current repro pipeline.

Input:
    - A labels CSV produced by build_external_labels_csv.py
    - Each row must contain at least:
        StudyInstanceUID, SeriesInstanceUID, source_path

Output layout:
    <output-root>/train/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

By default the script creates symlinks, so the original hospital data stays
untouched and no large DICOM files are duplicated.

Example:
    python reorganize_external_dicoms.py \
        --labels-csv ./external_labels.csv \
        --output-root /data/zilizhu/PE/CJ-EXTERNAL-STANDARD
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import pandas as pd
import pydicom


def is_hidden(name):
    return name.startswith(".")


def try_read_dicom(file_path):
    try:
        dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        if not hasattr(dataset, "StudyInstanceUID"):
            return None
        if not hasattr(dataset, "SeriesInstanceUID"):
            return None
        if not hasattr(dataset, "SOPInstanceUID"):
            return None
        return dataset
    except Exception:
        return None


def iter_visible_files(series_dir):
    for file_path in sorted(series_dir.iterdir()):
        if file_path.is_file() and not is_hidden(file_path.name):
            yield file_path


def link_or_copy(src_path, dst_path, mode):
    if dst_path.exists() or dst_path.is_symlink():
        if mode == "symlink" and dst_path.is_symlink():
            existing_target = os.path.realpath(dst_path)
            if existing_target == str(src_path.resolve()):
                return "already_linked"
        return "exists_conflict"

    if mode == "symlink":
        os.symlink(src_path.resolve(), dst_path)
    else:
        shutil.copy2(src_path, dst_path)
    return "created"


def process_series_row(row, output_train_dir, mode):
    expected_study_uid = str(row["StudyInstanceUID"])
    expected_series_uid = str(row["SeriesInstanceUID"])
    source_path = Path(row["source_path"]).expanduser()
    series_output_dir = output_train_dir / expected_study_uid / expected_series_uid
    series_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "StudyInstanceUID": expected_study_uid,
        "SeriesInstanceUID": expected_series_uid,
        "class_name": row.get("class_name", ""),
        "external_case_label": row.get("external_case_label", ""),
        "source_path": str(source_path),
        "num_source_files": 0,
        "num_readable_dicoms": 0,
        "num_created": 0,
        "num_already_linked": 0,
        "num_exists_conflict": 0,
        "num_uid_mismatch": 0,
        "num_unreadable_files": 0,
        "output_series_path": str(series_output_dir),
        "status": "ok",
    }

    if not source_path.exists():
        summary["status"] = "missing_source_path"
        return summary

    for file_path in iter_visible_files(source_path):
        summary["num_source_files"] += 1
        dataset = try_read_dicom(file_path)
        if dataset is None:
            summary["num_unreadable_files"] += 1
            continue

        study_uid = str(dataset.StudyInstanceUID)
        series_uid = str(dataset.SeriesInstanceUID)
        sop_uid = str(dataset.SOPInstanceUID)
        summary["num_readable_dicoms"] += 1

        if study_uid != expected_study_uid or series_uid != expected_series_uid:
            summary["num_uid_mismatch"] += 1
            continue

        dst_path = series_output_dir / f"{sop_uid}.dcm"
        result = link_or_copy(file_path, dst_path, mode=mode)
        if result == "created":
            summary["num_created"] += 1
        elif result == "already_linked":
            summary["num_already_linked"] += 1
        else:
            summary["num_exists_conflict"] += 1

    if summary["num_readable_dicoms"] == 0:
        summary["status"] = "no_readable_dicoms"
    elif summary["num_created"] == 0 and summary["num_already_linked"] == 0:
        summary["status"] = "no_output_created"
    elif summary["num_uid_mismatch"] > 0 or summary["num_exists_conflict"] > 0:
        summary["status"] = "warning"

    return summary


def write_summary_csv(summary_csv_path, rows):
    fieldnames = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "class_name",
        "external_case_label",
        "source_path",
        "num_source_files",
        "num_readable_dicoms",
        "num_created",
        "num_already_linked",
        "num_exists_conflict",
        "num_uid_mismatch",
        "num_unreadable_files",
        "output_series_path",
        "status",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Reorganize external DICOM files into RSNA-style Study/Series/SOP.dcm layout.")
    parser.add_argument("--labels-csv", required=True, help="Path to external_labels.csv or external_labels_dedup.csv.")
    parser.add_argument("--output-root", required=True, help="Root directory for standardized output.")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink", help="Use symlink by default to avoid duplicating data.")
    parser.add_argument("--split-name", default="train", help="Output split directory name. Default: train")
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_train_dir = output_root / args.split_name
    summary_csv_path = output_root / "reorganize_summary.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_csv}")

    df = pd.read_csv(labels_csv)
    required_columns = {"StudyInstanceUID", "SeriesInstanceUID", "source_path"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in labels CSV: {missing_columns}")

    before_rows = len(df)
    df = df.drop_duplicates(subset=["StudyInstanceUID", "SeriesInstanceUID"], keep="first").copy()
    after_rows = len(df)
    dropped_rows = before_rows - after_rows

    output_train_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for _, row in df.iterrows():
        summary_rows.append(process_series_row(row, output_train_dir=output_train_dir, mode=args.mode))

    write_summary_csv(summary_csv_path, summary_rows)

    total_created = sum(row["num_created"] for row in summary_rows)
    total_already_linked = sum(row["num_already_linked"] for row in summary_rows)
    total_mismatch = sum(row["num_uid_mismatch"] for row in summary_rows)
    total_conflict = sum(row["num_exists_conflict"] for row in summary_rows)
    warning_count = sum(1 for row in summary_rows if row["status"] == "warning")
    failed_count = sum(1 for row in summary_rows if row["status"] not in {"ok", "warning"})

    print(f"Input rows: {before_rows}")
    print(f"Rows after Study/Series dedup: {after_rows}")
    print(f"Dropped duplicate rows: {dropped_rows}")
    print(f"Output train dir: {output_train_dir}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Created links/files: {total_created}")
    print(f"Already linked: {total_already_linked}")
    print(f"UID mismatch files skipped: {total_mismatch}")
    print(f"Existing path conflicts: {total_conflict}")
    print(f"Series with warnings: {warning_count}")
    print(f"Series with failures: {failed_count}")


if __name__ == "__main__":
    main()
