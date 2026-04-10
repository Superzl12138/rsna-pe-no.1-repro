import numpy as np
import os


def _binary_confusion_counts(y_true, y_pred_binary):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred_binary = np.asarray(y_pred_binary).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    return tn, fp, fn, tp


def _roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tpr = np.r_[0.0, tps[threshold_idxs] / pos, 1.0]
    fpr = np.r_[0.0, fps[threshold_idxs] / neg, 1.0]
    return float(np.trapz(tpr, fpr))


def _average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def calculate_binary_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.float32)
    y_pred_binary = (y_pred >= threshold).astype(np.int32)

    tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred_binary)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    beta_sq = 4.0
    f2 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall) if (beta_sq * precision + recall) > 0 else 0.0
    auc_roc = _roc_auc_score(y_true, y_pred)
    auc_pr = _average_precision_score(y_true, y_pred)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    lr_positive = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float("inf")
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float("inf")
    youden_index = sensitivity + specificity - 1

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "f2_score": f2,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "ppv": ppv,
        "npv": npv,
        "lr_positive": lr_positive,
        "lr_negative": lr_negative,
        "youden_index": youden_index,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def get_label_thresholds(default_threshold=0.5):
    label_thresholds = {}
    for label_name in [
        "negative_exam_for_pe",
        "indeterminate",
        "chronic_pe",
        "acute_and_chronic_pe",
        "central_pe",
        "leftsided_pe",
        "rightsided_pe",
        "rv_lv_ratio_gte_1",
        "rv_lv_ratio_lt_1",
        "pe_present_on_image",
    ]:
        env_key = f"{label_name.upper()}_THRESHOLD"
        env_key = env_key.replace("-", "_")
        label_thresholds[label_name] = float(os.environ.get(env_key, default_threshold))
    return label_thresholds


def calculate_weighted_metrics(predictions_dict, labels_dict, loss_weight_dict, threshold=0.5, label_thresholds=None):
    all_metrics = {}
    weighted_metrics = {}
    if label_thresholds is None:
        label_thresholds = get_label_thresholds(default_threshold=threshold)
    for label_name, y_pred in predictions_dict.items():
        y_true = labels_dict.get(label_name, [])
        if len(y_pred) == 0 or len(y_true) == 0:
            continue
        all_metrics[label_name] = calculate_binary_metrics(
            y_true,
            y_pred,
            threshold=label_thresholds.get(label_name, threshold),
        )

    weighted_keys = ["auc_roc", "auc_pr", "f1_score", "f2_score", "sensitivity", "specificity"]
    if len(all_metrics) > 0:
        for key in weighted_keys:
            total_weight = 0.0
            weighted_sum = 0.0
            for label_name, metrics in all_metrics.items():
                weight = loss_weight_dict[label_name]
                weighted_sum += metrics[key] * weight
                total_weight += weight
            if total_weight > 0:
                weighted_metrics[f"weighted_{key}"] = weighted_sum / total_weight

        for key in weighted_keys:
            weighted_metrics[f"mean_{key}"] = float(np.mean([metrics[key] for metrics in all_metrics.values()]))

    pe_related_labels = ["leftsided_pe", "rightsided_pe", "central_pe", "chronic_pe", "acute_and_chronic_pe"]
    overall_true = []
    overall_pred = []
    if all(label in labels_dict and len(labels_dict[label]) > 0 for label in pe_related_labels):
        num_samples = len(labels_dict[pe_related_labels[0]])
        for i in range(num_samples):
            overall_true.append(max(labels_dict[label][i] for label in pe_related_labels))
            overall_pred.append(max(predictions_dict[label][i] for label in pe_related_labels))
        overall_pe_metrics = calculate_binary_metrics(overall_true, overall_pred, threshold=threshold)
    else:
        overall_pe_metrics = None

    return all_metrics, weighted_metrics, overall_pe_metrics


def print_validation_metrics(epoch, kaggle_loss, all_metrics, weighted_metrics, overall_pe_metrics):
    print()
    print("=" * 60)
    print(f"Epoch {epoch} Detailed Validation Metrics")
    print("=" * 60)
    print(f"kaggle_loss: {float(kaggle_loss):.6f}")

    if len(weighted_metrics) > 0:
        print("-" * 60)
        print("Weighted Metrics")
        for key in sorted(weighted_metrics.keys()):
            print(f"{key}: {weighted_metrics[key]:.6f}")

    if overall_pe_metrics is not None:
        print("-" * 60)
        print("Overall PE Detection")
        print(f"auc_roc: {overall_pe_metrics['auc_roc']:.6f}")
        print(f"auc_pr: {overall_pe_metrics['auc_pr']:.6f}")
        print(f"f1_score: {overall_pe_metrics['f1_score']:.6f}")
        print(f"f2_score: {overall_pe_metrics['f2_score']:.6f}")
        print(f"sensitivity: {overall_pe_metrics['sensitivity']:.6f}")
        print(f"specificity: {overall_pe_metrics['specificity']:.6f}")

    print("-" * 60)
    print("Per-label Metrics")
    for label_name in [
        "negative_exam_for_pe",
        "indeterminate",
        "chronic_pe",
        "acute_and_chronic_pe",
        "central_pe",
        "leftsided_pe",
        "rightsided_pe",
        "rv_lv_ratio_gte_1",
        "rv_lv_ratio_lt_1",
        "pe_present_on_image",
    ]:
        if label_name not in all_metrics:
            continue
        metrics = all_metrics[label_name]
        print(
            f"{label_name}: auc_roc={metrics['auc_roc']:.6f}, auc_pr={metrics['auc_pr']:.6f}, "
            f"f1={metrics['f1_score']:.6f}, f2={metrics['f2_score']:.6f}, "
            f"sensitivity={metrics['sensitivity']:.6f}, specificity={metrics['specificity']:.6f}, "
            f"ppv={metrics['ppv']:.6f}, npv={metrics['npv']:.6f}, threshold={metrics['threshold']:.3f}"
        )
    print("=" * 60)
    print()
