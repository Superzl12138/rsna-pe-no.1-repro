import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_DISPLAY = {
    "seresnext50_128": "SE-ResNeXt50 + 2nd-128",
    "seresnext50_192": "SE-ResNeXt50 + 2nd-192",
    "seresnext101_128": "SE-ResNeXt101 + 2nd-128",
    "seresnext101_192": "SE-ResNeXt101 + 2nd-192",
}

MODEL_ORDER = [
    "seresnext50_128",
    "seresnext50_192",
    "seresnext101_128",
    "seresnext101_192",
]

MODEL_COLORS = {
    "seresnext50_128": "#1f77b4",
    "seresnext50_192": "#ff7f0e",
    "seresnext101_128": "#2ca02c",
    "seresnext101_192": "#d62728",
}

TARGET_LABELS = [
    "negative_exam_for_pe",
    "chronic_pe",
    "acute_and_chronic_pe",
    "pe_present_on_image",
]

LABEL_DISPLAY = {
    "negative_exam_for_pe": "Negative exam for PE",
    "chronic_pe": "Chronic PE",
    "acute_and_chronic_pe": "Acute-and-chronic PE",
    "pe_present_on_image": "PE present on image",
}

PLOT_METRICS = ["auc_roc", "auc_pr", "f1", "f2"]
PLOT_METRIC_DISPLAY = {
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "f1": "F1 score",
    "f2": "F2 score",
}

BAR_METRICS = ["auc_roc", "auc_pr", "f1", "f2", "sensitivity", "specificity"]


def metric_key(metric_name):
    return {"f1": "f1", "f2": "f2"}.get(metric_name, metric_name)


def infer_model_key(path: Path):
    stem = path.stem
    stem = re.sub(r"\(\d+\)$", "", stem)
    return stem


def parse_log(path: Path):
    epoch_losses = {}
    epoch_label_metrics = {}
    current_epoch = None
    per_label_block = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            match_loss = re.search(r"epoch:\s*(\d+).+kaggle_loss:\s*([0-9.]+)", line)
            if match_loss:
                epoch = int(match_loss.group(1))
                epoch_losses[epoch] = float(match_loss.group(2))

            match_epoch = re.match(r"Epoch\s+(\d+)\s+Detailed Validation Metrics", line)
            if match_epoch:
                current_epoch = int(match_epoch.group(1))
                epoch_label_metrics.setdefault(current_epoch, {})
                per_label_block = False
                continue

            if line == "Per-label Metrics":
                per_label_block = True
                continue

            if per_label_block and line.startswith("===="):
                per_label_block = False
                continue

            if per_label_block and current_epoch is not None and ": auc_roc=" in line:
                label_name, metric_blob = line.split(": ", 1)
                metrics = {}
                for token in metric_blob.split(", "):
                    metric_name, metric_value = token.split("=")
                    metrics[metric_name] = float(metric_value)
                epoch_label_metrics[current_epoch][label_name] = metrics

    best_epoch = min(epoch_losses, key=epoch_losses.get) if epoch_losses else None
    best_kaggle_loss = epoch_losses[best_epoch] if best_epoch is not None else None
    return {
        "epoch_losses": epoch_losses,
        "epoch_label_metrics": epoch_label_metrics,
        "best_epoch": best_epoch,
        "best_kaggle_loss": best_kaggle_loss,
    }


def build_series(parsed_logs):
    series = {}
    for model_key, info in parsed_logs.items():
        series[model_key] = {}
        for label_name in TARGET_LABELS:
            series[model_key][label_name] = {}
            for metric_name in PLOT_METRICS:
                points = []
                for epoch in sorted(info["epoch_label_metrics"].keys()):
                    if label_name not in info["epoch_label_metrics"][epoch]:
                        continue
                    if metric_name not in info["epoch_label_metrics"][epoch][label_name]:
                        continue
                    points.append((epoch, info["epoch_label_metrics"][epoch][label_name][metric_name]))
                series[model_key][label_name][metric_name] = points
    return series


def plot_metric_curves(parsed_logs, output_dir: Path):
    series = build_series(parsed_logs)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(len(TARGET_LABELS), len(PLOT_METRICS), figsize=(18, 15), sharex=False, sharey=False)
    for row, label_name in enumerate(TARGET_LABELS):
        for col, metric_name in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            for model_key in MODEL_ORDER:
                points = series.get(model_key, {}).get(label_name, {}).get(metric_name, [])
                if len(points) == 0:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.plot(
                    xs,
                    ys,
                    label=MODEL_DISPLAY[model_key],
                    color=MODEL_COLORS[model_key],
                    linewidth=2.0,
                    marker="o",
                    markersize=3.5,
                )
            ax.set_title(f"{LABEL_DISPLAY[label_name]} | {PLOT_METRIC_DISPLAY[metric_name]}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(PLOT_METRIC_DISPLAY[metric_name])
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_ylim(0.0, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Validation trajectories of clinically relevant PE labels", fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "target_label_metric_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "target_label_metric_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_best_epoch_bars(parsed_logs, output_dir: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    width = 0.18
    x = np.arange(len(BAR_METRICS))

    for ax, label_name in zip(axes.flatten(), TARGET_LABELS):
        for idx, model_key in enumerate(MODEL_ORDER):
            best_epoch = parsed_logs[model_key]["best_epoch"]
            metrics = parsed_logs[model_key]["epoch_label_metrics"][best_epoch][label_name]
            values = []
            for metric_name in BAR_METRICS:
                values.append(metrics[metric_key(metric_name)])
            offset = (idx - 1.5) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLORS[model_key],
                label=MODEL_DISPLAY[model_key],
                alpha=0.9,
            )
        ax.set_title(f"{LABEL_DISPLAY[label_name]} | best epoch snapshot")
        ax.set_xticks(x)
        ax.set_xticklabels(["AUC-ROC", "AUC-PR", "F1", "F2", "Sens.", "Spec."], rotation=20)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Best-epoch metric comparison across four 2nd-level configurations", fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "target_label_best_epoch_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "target_label_best_epoch_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def export_summary_csv(parsed_logs, output_dir: Path):
    rows = []
    for model_key in MODEL_ORDER:
        best_epoch = parsed_logs[model_key]["best_epoch"]
        best_kaggle_loss = parsed_logs[model_key]["best_kaggle_loss"]
        for label_name in TARGET_LABELS:
            metrics = parsed_logs[model_key]["epoch_label_metrics"][best_epoch][label_name]
            rows.append({
                "model": model_key,
                "best_epoch": best_epoch,
                "best_kaggle_loss": best_kaggle_loss,
                "label": label_name,
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "f1_score": metrics["f1"],
                "f2_score": metrics["f2"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
            })
    fieldnames = list(rows[0].keys())
    with (output_dir / "target_label_best_epoch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_paths(log_dir: Path, filenames):
    resolved = []
    for filename in filenames:
        path = log_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing log file: {path}")
        resolved.append(path)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("/Users/bytedance/Desktop/project/RSNA-STR-Pulmonary-Embolism-Detection/repro/logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/bytedance/Desktop/project/RSNA-STR-Pulmonary-Embolism-Detection/repro/paper_figures/2nd_level_t1000v100"))
    parser.add_argument(
        "--files",
        nargs=4,
        default=[
            "seresnext50_128(1).txt",
            "seresnext50_192(1).txt",
            "seresnext101_128(1).txt",
            "seresnext101_192(1).txt",
        ],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_logs = {}
    for path in resolve_paths(args.log_dir, args.files):
        model_key = infer_model_key(path)
        parsed_logs[model_key] = parse_log(path)

    plot_metric_curves(parsed_logs, args.output_dir)
    plot_best_epoch_bars(parsed_logs, args.output_dir)
    export_summary_csv(parsed_logs, args.output_dir)

    print("figures saved to", args.output_dir)
    for model_key in MODEL_ORDER:
        print(model_key, parsed_logs[model_key]["best_epoch"], parsed_logs[model_key]["best_kaggle_loss"])


if __name__ == "__main__":
    main()
