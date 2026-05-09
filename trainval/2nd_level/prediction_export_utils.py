import csv
import gzip
import os

from metrics import get_label_thresholds


EXAM_LABELS = [
    "negative_exam_for_pe",
    "indeterminate",
    "chronic_pe",
    "acute_and_chronic_pe",
    "central_pe",
    "leftsided_pe",
    "rightsided_pe",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
]

LABELS = EXAM_LABELS + ["pe_present_on_image"]


def init_validation_sample_metadata():
    return {
        "sample_id": {label_name: [] for label_name in LABELS},
        "series_id": {label_name: [] for label_name in LABELS},
        "image_id": {label_name: [] for label_name in LABELS},
        "sample_level": {label_name: [] for label_name in LABELS},
    }


def append_exam_sample_metadata(sample_metadata, series_id):
    for label_name in EXAM_LABELS:
        sample_metadata["sample_id"][label_name].append(series_id)
        sample_metadata["series_id"][label_name].append(series_id)
        sample_metadata["image_id"][label_name].append("")
        sample_metadata["sample_level"][label_name].append("exam")


def append_image_sample_metadata(sample_metadata, series_id, image_ids):
    for image_id in image_ids:
        sample_metadata["sample_id"]["pe_present_on_image"].append(image_id)
        sample_metadata["series_id"]["pe_present_on_image"].append(series_id)
        sample_metadata["image_id"]["pe_present_on_image"].append(image_id)
        sample_metadata["sample_level"]["pe_present_on_image"].append("image")


def _prediction_export_enabled():
    return os.environ.get("EXPORT_VALID_PREDICTIONS", "0").lower() in {"1", "true", "yes", "y"}


def _should_export_epoch(epoch, num_epoch):
    mode = os.environ.get("VALID_PREDICTION_EXPORT_MODE", "all").lower()
    if mode in {"all", "every"}:
        return True
    if mode in {"last", "final"}:
        return epoch == num_epoch - 1
    if mode == "interval":
        interval = max(1, int(os.environ.get("VALID_PREDICTION_EXPORT_INTERVAL", "1")))
        return epoch % interval == 0 or epoch == num_epoch - 1
    if mode == "none":
        return False
    return True


def export_validation_predictions_csv(
    output_dir,
    experiment_name,
    model_name,
    epoch,
    num_epoch,
    predictions_dict,
    labels_dict,
    sample_metadata,
):
    if not _prediction_export_enabled() or not _should_export_epoch(epoch, num_epoch):
        return None

    experiment_name = experiment_name or os.environ.get("EXPERIMENT_NAME", "")
    if not experiment_name:
        experiment_name = model_name

    output_dir = os.environ.get("VALID_PREDICTION_EXPORT_DIR", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{experiment_name}_{model_name}_epoch{epoch:03d}_valid_predictions.csv.gz",
    )
    thresholds = get_label_thresholds(default_threshold=0.5)

    fieldnames = [
        "experiment",
        "model",
        "epoch",
        "sample_level",
        "sample_id",
        "series_id",
        "image_id",
        "label",
        "y_true",
        "y_pred",
        "threshold",
    ]

    with gzip.open(output_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label_name in LABELS:
            y_pred_values = predictions_dict.get(label_name, [])
            y_true_values = labels_dict.get(label_name, [])
            sample_ids = sample_metadata["sample_id"].get(label_name, [])
            series_ids = sample_metadata["series_id"].get(label_name, [])
            image_ids = sample_metadata["image_id"].get(label_name, [])
            sample_levels = sample_metadata["sample_level"].get(label_name, [])
            if not (
                len(y_pred_values)
                == len(y_true_values)
                == len(sample_ids)
                == len(series_ids)
                == len(image_ids)
                == len(sample_levels)
            ):
                raise ValueError(
                    f"Prediction export length mismatch for {label_name}: "
                    f"pred={len(y_pred_values)}, true={len(y_true_values)}, "
                    f"sample_id={len(sample_ids)}"
                )
            for i in range(len(y_pred_values)):
                writer.writerow(
                    {
                        "experiment": experiment_name,
                        "model": model_name,
                        "epoch": epoch,
                        "sample_level": sample_levels[i],
                        "sample_id": sample_ids[i],
                        "series_id": series_ids[i],
                        "image_id": image_ids[i],
                        "label": label_name,
                        "y_true": float(y_true_values[i]),
                        "y_pred": float(y_pred_values[i]),
                        "threshold": thresholds.get(label_name, 0.5),
                    }
                )

    print(f"Saved validation predictions: {output_path}", flush=True)
    return output_path
