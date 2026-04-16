import os
from pathlib import Path

import cv2
import numpy as np
import pydicom

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = REPO_ROOT / "input" / "train"
DEFAULT_FRANGI_ROOT = Path("/data/zilizhu/PE/frangi_enhanced_512x512")


def window(img, WL=50, WW=350):
    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(img.copy(), lower, upper)
    x = x - np.min(x)
    max_value = np.max(x)
    if max_value > 0:
        x = x / max_value
    x = (x * 255.0).astype("uint8")
    return x


def get_frangi_config():
    enabled = os.environ.get("ENABLE_FRANGI", "0") == "1"
    mode = os.environ.get("FRANGI_MODE", "triplet_blend")
    if mode != "triplet_blend":
        raise ValueError(f"Unsupported FRANGI_MODE: {mode}")

    missing_policy = os.environ.get("FRANGI_MISSING_POLICY", "fallback")
    if missing_policy != "fallback":
        raise ValueError(f"Unsupported FRANGI_MISSING_POLICY: {missing_policy}")

    blend_alpha = float(os.environ.get("FRANGI_BLEND_ALPHA", "0.7"))
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError(f"FRANGI_BLEND_ALPHA must be within [0, 1], got {blend_alpha}")

    train_root = Path(os.environ.get("PE_TRAIN_DIR", str(DEFAULT_TRAIN_ROOT)))
    frangi_root = Path(os.environ.get("PE_FRANGI_DIR", str(DEFAULT_FRANGI_ROOT)))
    return {
        "enabled": enabled,
        "mode": mode,
        "missing_policy": missing_policy,
        "blend_alpha": blend_alpha,
        "train_root": train_root,
        "frangi_root": frangi_root,
    }


def print_frangi_config(config):
    print("=" * 60)
    print("Frangi Input")
    print("=" * 60)
    print(f"enabled: {config['enabled']}")
    print(f"mode: {config['mode']}")
    print(f"missing_policy: {config['missing_policy']}")
    print(f"blend_alpha: {config['blend_alpha']:.3f}")
    print(f"train_root: {config['train_root']}")
    print(f"frangi_root: {config['frangi_root']}")
    print("=" * 60)


def _series_dir(image_dict, image_id, train_root):
    study_id, series_id = image_dict[image_id]["series_id"].split("_")
    return train_root / study_id / series_id


def _read_hu_image(image_dict, image_id, train_root):
    dicom_path = _series_dir(image_dict, image_id, train_root) / f"{image_id}.dcm"
    data = pydicom.dcmread(str(dicom_path))
    img = data.pixel_array.astype(np.float32)
    img = img * float(data.RescaleSlope) + float(data.RescaleIntercept)
    return img


def _normalize_frangi_to_uint8(frangi_img):
    frangi_img = np.asarray(frangi_img).squeeze()
    if frangi_img.ndim != 2:
        raise ValueError(f"Expected 2D Frangi array, got shape {frangi_img.shape}")
    frangi_img = frangi_img.astype(np.float32)
    min_value = float(np.min(frangi_img))
    max_value = float(np.max(frangi_img))
    if max_value > min_value:
        frangi_img = (frangi_img - min_value) / (max_value - min_value)
    else:
        frangi_img = np.zeros_like(frangi_img, dtype=np.float32)
    return (frangi_img * 255.0).astype("uint8")


def _read_frangi_image(frangi_root, image_id):
    frangi_path = frangi_root / f"{image_id}.npy"
    if not frangi_path.exists():
        return None
    return _normalize_frangi_to_uint8(np.load(frangi_path))


def _blend_channel(raw_channel, frangi_channel, alpha):
    if frangi_channel is None:
        return raw_channel
    if frangi_channel.shape != raw_channel.shape:
        frangi_channel = cv2.resize(
            frangi_channel,
            (raw_channel.shape[1], raw_channel.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    blended = (
        alpha * raw_channel.astype(np.float32)
        + (1.0 - alpha) * frangi_channel.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype("uint8")


def build_image_triplet(image_dict, bbox_dict, center_image_id, target_size, frangi_config):
    image_info = image_dict[center_image_id]
    triplet_ids = [
        image_info["image_minus1"],
        center_image_id,
        image_info["image_plus1"],
    ]

    channels = []
    for image_id in triplet_ids:
        raw_channel = window(
            _read_hu_image(
                image_dict=image_dict,
                image_id=image_id,
                train_root=frangi_config["train_root"],
            ),
            WL=100,
            WW=700,
        )
        if frangi_config["enabled"]:
            frangi_channel = _read_frangi_image(frangi_config["frangi_root"], image_id)
            raw_channel = _blend_channel(
                raw_channel=raw_channel,
                frangi_channel=frangi_channel,
                alpha=frangi_config["blend_alpha"],
            )
        channels.append(raw_channel)

    x = np.stack(channels, axis=2)
    bbox = [int(v) for v in bbox_dict[image_info["series_id"]]]
    x = x[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
    x = cv2.resize(x, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return x
