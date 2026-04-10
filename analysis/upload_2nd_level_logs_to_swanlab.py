import argparse
from pathlib import Path

from plot_2nd_level_metrics import MODEL_ORDER, TARGET_LABELS, infer_model_key, parse_log, resolve_paths

try:
    import swanlab
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install swanlab first: pip install swanlab") from exc


WEIGHTED_KEYS = [
    "weighted_auc_roc",
    "weighted_auc_pr",
    "weighted_f1_score",
    "weighted_f2_score",
    "weighted_sensitivity",
    "weighted_specificity",
]


def parse_epoch_blocks(path: Path):
    blocks = {}
    current_epoch = None
    current_section = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("Epoch ") and "Detailed Validation Metrics" in line:
                current_epoch = int(line.split()[1])
                blocks[current_epoch] = {"weighted": {}, "overall": {}, "labels": {}}
                current_section = None
                continue
            if current_epoch is None:
                continue
            if line == "Weighted Metrics":
                current_section = "weighted"
                continue
            if line == "Overall PE Detection":
                current_section = "overall"
                continue
            if line == "Per-label Metrics":
                current_section = "labels"
                continue
            if line.startswith("-") or line.startswith("=") or line == "":
                continue
            if current_section in ("weighted", "overall") and ": " in line:
                key, value = line.split(": ", 1)
                blocks[current_epoch][current_section][key] = float(value)
            elif current_section == "labels" and ": auc_roc=" in line:
                label_name, metric_blob = line.split(": ", 1)
                metrics = {}
                for token in metric_blob.split(", "):
                    metric_name, metric_value = token.split("=")
                    metrics[metric_name] = float(metric_value)
                blocks[current_epoch]["labels"][label_name] = metrics
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--project", type=str, default="rsna-pe-repro")
    parser.add_argument("--experiment-name", type=str, default="trainval-2nd-level")
    parser.add_argument(
        "--files",
        nargs=4,
        default=[
            "seresnext50_128.txt",
            "seresnext50_192.txt",
            "seresnext101_128.txt",
            "seresnext101_192.txt",
        ],
    )
    args = parser.parse_args()

    resolved_paths = resolve_paths(args.log_dir, args.files)
    parsed = {}
    epoch_blocks = {}
    for path in resolved_paths:
        model_key = infer_model_key(path)
        parsed[model_key] = parse_log(path)
        epoch_blocks[model_key] = parse_epoch_blocks(path)

    config = {
        "log_dir": str(args.log_dir),
        "models": MODEL_ORDER,
        "target_labels": TARGET_LABELS,
    }
    swanlab.init(project=args.project, experiment_name=args.experiment_name, config=config)

    for model_key in MODEL_ORDER:
        best_epoch = parsed[model_key]["best_epoch"]
        best_kaggle_loss = parsed[model_key]["best_kaggle_loss"]
        swanlab.log(
            {
                f"{model_key}/best_epoch": best_epoch,
                f"{model_key}/best_kaggle_loss": best_kaggle_loss,
            }
        )
        for epoch in sorted(epoch_blocks[model_key].keys()):
            payload = {f"{model_key}/epoch": epoch}
            for key in WEIGHTED_KEYS:
                if key in epoch_blocks[model_key][epoch]["weighted"]:
                    payload[f"{model_key}/{key}"] = epoch_blocks[model_key][epoch]["weighted"][key]
            overall = epoch_blocks[model_key][epoch]["overall"]
            for key in ("auc_roc", "auc_pr", "f1_score", "f2_score", "sensitivity", "specificity"):
                if key in overall:
                    payload[f"{model_key}/overall_pe_{key}"] = overall[key]
            for label_name in TARGET_LABELS:
                metrics = epoch_blocks[model_key][epoch]["labels"].get(label_name, {})
                for key in ("auc_roc", "auc_pr", "f1", "f2", "sensitivity", "specificity"):
                    if key in metrics:
                        payload[f"{model_key}/{label_name}/{key}"] = metrics[key]
            swanlab.log(payload, step=epoch)

    swanlab.finish()
    print("Uploaded logs to SwanLab:", args.experiment_name)


if __name__ == "__main__":
    main()
