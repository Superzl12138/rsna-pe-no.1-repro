"""
Infer lung bounding boxes for external hospital series using the trained
EfficientNet lung localization model.

Unlike the original RSNA validation script, this version does not depend on
lung_bbox.csv. For each external series it uniformly samples 4 slices across the
sorted series, predicts 4 normalized boxes, and merges them into one
series-level bbox in 512-space.

Example:
    python infer_external_lung_bbox.py \
        --pickles-dir /data/zilizhu/PE/CJ-EXTERNAL-STANDARD/external_pickles \
        --train-root /data/zilizhu/PE/CJ-EXTERNAL-STANDARD/train \
        --weights-path ../trainval/lung_localization/split2/weights/epoch34_polyak \
        --output-dir /data/zilizhu/PE/CJ-EXTERNAL-STANDARD/external_bbox
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    fallback_dir = REPO_ROOT / "trainval" / "lung_localization" / "split2"
    if str(fallback_dir) not in sys.path:
        sys.path.insert(0, str(fallback_dir))
    from efficientnet_pytorch import EfficientNet


def window(x, wl=50, ww=350):
    upper, lower = wl + ww // 2, wl - ww // 2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    max_value = np.max(x)
    if max_value > 0:
        x = x / max_value
    return x


def select_uniform_image_list(sorted_image_list, num_images=4):
    if len(sorted_image_list) == 0:
        return []
    if len(sorted_image_list) >= num_images:
        selected_idx = np.linspace(0, len(sorted_image_list) - 1, num_images)
        return [sorted_image_list[int(round(i))] for i in selected_idx]
    selected = list(sorted_image_list)
    while len(selected) < num_images:
        selected.append(sorted_image_list[-1])
    return selected


class BboxDataset(Dataset):
    def __init__(self, series_list):
        self.series_list = series_list

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, index):
        return index


class BboxCollator:
    def __init__(self, series_list, series_dict, train_root):
        self.series_list = series_list
        self.series_dict = series_dict
        self.train_root = Path(train_root)

    def _read_selected_dicoms(self, series_dir, selected_image_list):
        dicoms = []
        for image_id in selected_image_list:
            dicom_path = series_dir / f"{image_id}.dcm"
            dataset = pydicom.dcmread(str(dicom_path), force=True)
            img = dataset.pixel_array.astype(np.float32)
            img = img * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
            dicoms.append(img)
        dicoms = np.asarray(dicoms, dtype=np.float32)
        return window(dicoms, wl=100, ww=700)

    def __call__(self, batch_idx):
        series_id = self.series_list[batch_idx[0]]
        study_id, series_uid = series_id.split("_", 1)
        series_dir = self.train_root / study_id / series_uid
        sorted_image_list = self.series_dict[series_id]["sorted_image_list"]
        selected_image_list = select_uniform_image_list(sorted_image_list, num_images=4)
        if len(selected_image_list) == 0:
            raise RuntimeError(f"No images found for {series_id}")

        dicoms = self._read_selected_dicoms(series_dir, selected_image_list)
        x = np.zeros((4, 3, dicoms.shape[1], dicoms.shape[2]), dtype=np.float32)
        for i in range(4):
            x[i, 0] = dicoms[i]
            x[i, 1] = dicoms[i]
            x[i, 2] = dicoms[i]
        return torch.from_numpy(x), selected_image_list, series_id


class EfficientNetBbox(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b0")
        in_features = self.net._fc.in_features
        self.last_linear = nn.Linear(in_features, 4)

    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


def write_summary(summary_path, rows):
    fieldnames = [
        "series_id",
        "selected_image_ids",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Infer external lung bboxes using trained EfficientNet weights.")
    parser.add_argument("--pickles-dir", required=True, help="Directory containing external pickle files.")
    parser.add_argument("--train-root", required=True, help="Standardized train root, e.g. /data/.../CJ-EXTERNAL-STANDARD/train")
    parser.add_argument("--weights-path", required=True, help="Path to trained lung localization weights, e.g. split2/weights/epoch34_polyak")
    parser.add_argument("--output-dir", required=True, help="Directory to save bbox_dict_external.pickle and summary CSV.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Inference device, default: cuda")
    args = parser.parse_args()

    pickles_dir = Path(args.pickles_dir).expanduser().resolve()
    train_root = Path(args.train_root).expanduser().resolve()
    weights_path = Path(args.weights_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not pickles_dir.exists():
        raise FileNotFoundError(f"Pickles dir does not exist: {pickles_dir}")
    if not train_root.exists():
        raise FileNotFoundError(f"Train root does not exist: {train_root}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path does not exist: {weights_path}")

    with open(pickles_dir / "series_dict.pickle", "rb") as handle:
        series_dict = pickle.load(handle)
    with open(pickles_dir / "series_list_external.pickle", "rb") as handle:
        series_list = pickle.load(handle)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = EfficientNetBbox()
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    datagen = BboxDataset(series_list=series_list)
    collate_fn = BboxCollator(series_list=series_list, series_dict=series_dict, train_root=train_root)
    generator = DataLoader(
        dataset=datagen,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    bbox_dict_external = {}
    summary_rows = []

    for images, selected_image_list, series_id in tqdm(generator, total=len(generator)):
        with torch.no_grad():
            images = images.to(device)
            logits = model(images)
            bbox = np.squeeze(logits.detach().cpu().numpy())

        xmin = np.round(min(bbox[:, 0]) * 512)
        ymin = np.round(min(bbox[:, 1]) * 512)
        xmax = np.round(max(bbox[:, 2]) * 512)
        ymax = np.round(max(bbox[:, 3]) * 512)
        merged_bbox = [
            int(max(0, xmin)),
            int(max(0, ymin)),
            int(min(512, xmax)),
            int(min(512, ymax)),
        ]
        bbox_dict_external[series_id] = merged_bbox
        summary_rows.append(
            {
                "series_id": series_id,
                "selected_image_ids": "|".join(selected_image_list),
                "bbox_xmin": merged_bbox[0],
                "bbox_ymin": merged_bbox[1],
                "bbox_xmax": merged_bbox[2],
                "bbox_ymax": merged_bbox[3],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_path = output_dir / "bbox_dict_external.pickle"
    summary_path = output_dir / "bbox_inference_summary.csv"
    with open(bbox_path, "wb") as handle:
        pickle.dump(bbox_dict_external, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_summary(summary_path, summary_rows)

    print(f"Series processed: {len(series_list)}")
    print(f"bbox_dict size: {len(bbox_dict_external)}")
    print(f"Weights path: {weights_path}")
    print(f"Output bbox pickle: {bbox_path}")
    print(f"Output summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
