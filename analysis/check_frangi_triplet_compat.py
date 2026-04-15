#!/usr/bin/env python3
import argparse
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = REPO_ROOT / "trainval" / "process_input" / "split2"
DEFAULT_BBOX_PATH = REPO_ROOT / "trainval" / "lung_localization" / "split2" / "bbox_dict_train.pickle"
DEFAULT_TRAIN_ROOT = Path("/data/zilizhu/PE/train")
DEFAULT_FRANGI_ROOT = Path("/data/zilizhu/PE/frangi_enhanced_512x512")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_triplet(image_dict, image_id):
    image_info = image_dict[image_id]
    return [
        image_info["image_minus1"],
        image_id,
        image_info["image_plus1"],
    ]


def dicom_path_from_image_id(train_root, image_dict, image_id):
    series_key = image_dict[image_id]["series_id"]
    study_id, series_id = series_key.split("_")
    return train_root / study_id / series_id / f"{image_id}.dcm"


def frangi_path_from_image_id(frangi_root, image_id):
    return frangi_root / f"{image_id}.npy"


def read_dicom_shape(dicom_path):
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=False)
    return tuple(int(v) for v in ds.pixel_array.shape)


def summarize_counter(counter, top_k=10):
    return ", ".join([f"{k}:{v}" for k, v in counter.most_common(top_k)]) if counter else "none"


def main():
    parser = argparse.ArgumentParser(description="Check whether Frangi triplet inputs are compatible with current repro image pipeline.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory containing image_list/image_dict pickle files.",
    )
    parser.add_argument(
        "--bbox-path",
        type=Path,
        default=DEFAULT_BBOX_PATH,
        help="bbox_dict pickle path used by current repro image-level training.",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path(os.environ.get("PE_TRAIN_DIR", str(DEFAULT_TRAIN_ROOT))),
        help="Root directory of train DICOM files.",
    )
    parser.add_argument(
        "--frangi-root",
        type=Path,
        default=Path(os.environ.get("PE_FRANGI_DIR", str(DEFAULT_FRANGI_ROOT))),
        help="Root directory of precomputed Frangi .npy files.",
    )
    parser.add_argument(
        "--image-list-name",
        type=str,
        default="image_list_train.pickle",
        help="Which image list pickle to inspect.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="How many center image_ids to inspect. Use 0 to inspect all.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start offset inside image list.",
    )
    args = parser.parse_args()

    image_list = load_pickle(args.split_dir / args.image_list_name)
    image_dict = load_pickle(args.split_dir / "image_dict.pickle")
    bbox_dict = load_pickle(args.bbox_path)
    frangi_root = args.frangi_root
    train_root = args.train_root

    if args.offset:
        image_list = image_list[args.offset:]
    if args.sample_limit > 0:
        image_list = image_list[:args.sample_limit]

    stats = {
        "centers_checked": 0,
        "triplets_complete": 0,
        "triplets_missing_frangi": 0,
        "triplets_missing_dicom": 0,
        "triplets_missing_bbox": 0,
        "triplets_same_frangi_shape": 0,
        "triplets_same_dicom_shape": 0,
        "triplets_direct_bbox_compatible_with_frangi": 0,
        "triplets_scaled_bbox_compatible_with_frangi": 0,
    }
    missing_frangi_examples = []
    missing_dicom_examples = []
    missing_bbox_examples = []
    dicom_shape_counter = Counter()
    frangi_shape_counter = Counter()
    frangi_per_position_shape_counter = {
        "prev": Counter(),
        "curr": Counter(),
        "next": Counter(),
    }
    direct_shape_match_counter = Counter()

    for center_image_id in image_list:
        stats["centers_checked"] += 1
        series_id = image_dict[center_image_id]["series_id"]
        if series_id not in bbox_dict:
            stats["triplets_missing_bbox"] += 1
            if len(missing_bbox_examples) < 10:
                missing_bbox_examples.append((center_image_id, series_id))
            continue

        triplet_ids = resolve_triplet(image_dict, center_image_id)
        bbox = bbox_dict[series_id]
        bbox = [int(v) for v in bbox]

        dicom_shapes = []
        frangi_shapes = []
        triplet_ok = True

        for position_name, image_id in zip(["prev", "curr", "next"], triplet_ids):
            dicom_path = dicom_path_from_image_id(train_root, image_dict, image_id)
            frangi_path = frangi_path_from_image_id(frangi_root, image_id)

            if not dicom_path.exists():
                stats["triplets_missing_dicom"] += 1
                if len(missing_dicom_examples) < 10:
                    missing_dicom_examples.append((position_name, image_id, str(dicom_path)))
                triplet_ok = False
                break
            if not frangi_path.exists():
                stats["triplets_missing_frangi"] += 1
                if len(missing_frangi_examples) < 10:
                    missing_frangi_examples.append((position_name, image_id, str(frangi_path)))
                triplet_ok = False
                break

            dicom_shape = read_dicom_shape(dicom_path)
            frangi_shape = tuple(int(v) for v in np.load(frangi_path, mmap_mode="r").shape)
            dicom_shapes.append(dicom_shape)
            frangi_shapes.append(frangi_shape)

            dicom_shape_counter[dicom_shape] += 1
            frangi_shape_counter[frangi_shape] += 1
            frangi_per_position_shape_counter[position_name][frangi_shape] += 1
            direct_shape_match_counter[(dicom_shape == frangi_shape)] += 1

        if not triplet_ok:
            continue

        stats["triplets_complete"] += 1
        if len(set(dicom_shapes)) == 1:
            stats["triplets_same_dicom_shape"] += 1
        if len(set(frangi_shapes)) == 1:
            stats["triplets_same_frangi_shape"] += 1

        direct_bbox_ok = True
        scaled_bbox_ok = True
        for dicom_shape, frangi_shape in zip(dicom_shapes, frangi_shapes):
            h, w = dicom_shape
            fh, fw = frangi_shape
            x1, y1, x2, y2 = bbox

            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                direct_bbox_ok = False
                scaled_bbox_ok = False
                break

            if not (x2 <= fw and y2 <= fh):
                direct_bbox_ok = False

            sx1 = int(np.floor(x1 * fw / w))
            sx2 = int(np.ceil(x2 * fw / w))
            sy1 = int(np.floor(y1 * fh / h))
            sy2 = int(np.ceil(y2 * fh / h))
            if not (0 <= sx1 < sx2 <= fw and 0 <= sy1 < sy2 <= fh):
                scaled_bbox_ok = False

        if direct_bbox_ok:
            stats["triplets_direct_bbox_compatible_with_frangi"] += 1
        if scaled_bbox_ok:
            stats["triplets_scaled_bbox_compatible_with_frangi"] += 1

    print("=" * 80)
    print("Frangi Triplet Compatibility Report")
    print("=" * 80)
    print(f"split_dir: {args.split_dir}")
    print(f"bbox_path: {args.bbox_path}")
    print(f"train_root: {train_root}")
    print(f"frangi_root: {frangi_root}")
    print(f"image_list_name: {args.image_list_name}")
    print(f"centers_checked: {stats['centers_checked']}")
    print(f"triplets_complete: {stats['triplets_complete']}")
    print(f"triplets_missing_bbox: {stats['triplets_missing_bbox']}")
    print(f"triplets_missing_dicom: {stats['triplets_missing_dicom']}")
    print(f"triplets_missing_frangi: {stats['triplets_missing_frangi']}")
    print(f"triplets_same_dicom_shape: {stats['triplets_same_dicom_shape']}")
    print(f"triplets_same_frangi_shape: {stats['triplets_same_frangi_shape']}")
    print(f"triplets_direct_bbox_compatible_with_frangi: {stats['triplets_direct_bbox_compatible_with_frangi']}")
    print(f"triplets_scaled_bbox_compatible_with_frangi: {stats['triplets_scaled_bbox_compatible_with_frangi']}")
    print("-" * 80)
    print(f"top_dicom_shapes: {summarize_counter(dicom_shape_counter)}")
    print(f"top_frangi_shapes: {summarize_counter(frangi_shape_counter)}")
    print(f"direct_shape_match_counts: {dict(direct_shape_match_counter)}")
    print(f"prev_frangi_shapes: {summarize_counter(frangi_per_position_shape_counter['prev'])}")
    print(f"curr_frangi_shapes: {summarize_counter(frangi_per_position_shape_counter['curr'])}")
    print(f"next_frangi_shapes: {summarize_counter(frangi_per_position_shape_counter['next'])}")
    print("-" * 80)
    print(f"missing_bbox_examples: {missing_bbox_examples}")
    print(f"missing_dicom_examples: {missing_dicom_examples}")
    print(f"missing_frangi_examples: {missing_frangi_examples}")
    print("=" * 80)

    if stats["triplets_complete"] == 0:
        print("FINAL_JUDGEMENT: no complete triplets found; cannot assess compatibility.")
        return

    if stats["triplets_direct_bbox_compatible_with_frangi"] == stats["triplets_complete"]:
        print("FINAL_JUDGEMENT: Frangi triplets appear directly compatible with current bbox coordinates.")
    elif stats["triplets_scaled_bbox_compatible_with_frangi"] == stats["triplets_complete"]:
        print("FINAL_JUDGEMENT: Frangi triplets do not share raw DICOM coordinates, but scaled bbox mapping looks fully compatible.")
    else:
        print("FINAL_JUDGEMENT: Frangi triplets are only partially compatible; inspect missing files or coordinate mismatch before integration.")


if __name__ == "__main__":
    main()
