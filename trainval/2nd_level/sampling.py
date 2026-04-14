import os

import numpy as np


def _parse_env_float(env_key, default, minimum=None, maximum=None):
    value = os.environ.get(env_key, str(default))
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"Invalid float value for {env_key}: {value}")
    if minimum is not None and parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _classify_series(series_id, series_dict):
    is_chronic = int(series_dict[series_id]["chronic_pe"]) == 1
    is_acute_and_chronic = int(series_dict[series_id]["acute_and_chronic_pe"]) == 1
    if is_acute_and_chronic:
        return "acute_and_chronic"
    if is_chronic:
        return "chronic_only"
    return "other"


def _summarize_series_list(series_list, series_dict):
    summary = {
        "total": len(series_list),
        "chronic_positive": 0,
        "acute_and_chronic_positive": 0,
        "chronic_only": 0,
        "other_series": 0,
    }
    for series_id in series_list:
        series_group = _classify_series(series_id, series_dict)
        if series_group in {"chronic_only", "acute_and_chronic"}:
            summary["chronic_positive"] += 1
        if series_group == "acute_and_chronic":
            summary["acute_and_chronic_positive"] += 1
        elif series_group == "chronic_only":
            summary["chronic_only"] += 1
        else:
            summary["other_series"] += 1
    return summary


def _apply_undersampling(series_list, series_dict, rng, keep_other_ratio):
    filtered_series_list = []
    for series_id in series_list:
        series_group = _classify_series(series_id, series_dict)
        if series_group != "other":
            filtered_series_list.append(series_id)
            continue
        if rng.rand() < keep_other_ratio:
            filtered_series_list.append(series_id)
    return filtered_series_list


def _apply_oversampling(series_list, series_dict, rng, chronic_factor, acute_and_chronic_factor):
    expanded_series_list = []
    for series_id in series_list:
        series_group = _classify_series(series_id, series_dict)
        target_factor = 1.0
        if series_group in {"chronic_only", "acute_and_chronic"}:
            target_factor = max(target_factor, chronic_factor)
        if series_group == "acute_and_chronic":
            target_factor = max(target_factor, acute_and_chronic_factor)

        repeat_count = int(np.floor(target_factor))
        fractional = target_factor - repeat_count
        if rng.rand() < fractional:
            repeat_count += 1
        expanded_series_list.extend([series_id] * repeat_count)
    return expanded_series_list


def build_resampled_series_list(series_list, series_dict, seed):
    oversampling_enabled = os.environ.get("ENABLE_2ND_LEVEL_OVERSAMPLING", "0") == "1"
    undersampling_enabled = os.environ.get("ENABLE_2ND_LEVEL_UNDERSAMPLING", "0") == "1"
    chronic_factor = _parse_env_float("OVERSAMPLE_CHRONIC_PE_FACTOR", default=1.0, minimum=1.0)
    acute_and_chronic_factor = _parse_env_float(
        "OVERSAMPLE_ACUTE_AND_CHRONIC_PE_FACTOR",
        default=1.0,
        minimum=1.0,
    )
    keep_other_ratio = _parse_env_float(
        "UNDERSAMPLE_OTHER_SERIES_RATIO",
        default=1.0,
        minimum=0.0,
        maximum=1.0,
    )

    original_summary = _summarize_series_list(series_list, series_dict)
    rng = np.random.RandomState(seed)

    undersampled_series_list = list(series_list)
    if undersampling_enabled:
        undersampled_series_list = _apply_undersampling(
            series_list=undersampled_series_list,
            series_dict=series_dict,
            rng=rng,
            keep_other_ratio=keep_other_ratio,
        )

    resampled_series_list = list(undersampled_series_list)
    if oversampling_enabled:
        resampled_series_list = _apply_oversampling(
            series_list=resampled_series_list,
            series_dict=series_dict,
            rng=rng,
            chronic_factor=chronic_factor,
            acute_and_chronic_factor=acute_and_chronic_factor,
        )

    return resampled_series_list, {
        "oversampling_enabled": oversampling_enabled,
        "undersampling_enabled": undersampling_enabled,
        "chronic_factor": chronic_factor,
        "acute_and_chronic_factor": acute_and_chronic_factor,
        "keep_other_ratio": keep_other_ratio,
        "before": original_summary,
        "after_undersampling": _summarize_series_list(undersampled_series_list, series_dict),
        "after": _summarize_series_list(resampled_series_list, series_dict),
    }


def print_sampling_summary(summary):
    print("=" * 60)
    print("2nd-level Resampling")
    print("=" * 60)
    print(f"undersampling_enabled: {summary['undersampling_enabled']}")
    print(f"UNDERSAMPLE_OTHER_SERIES_RATIO: {summary['keep_other_ratio']:.3f}")
    print(f"oversampling_enabled: {summary['oversampling_enabled']}")
    print(f"OVERSAMPLE_CHRONIC_PE_FACTOR: {summary['chronic_factor']:.3f}")
    print(
        "OVERSAMPLE_ACUTE_AND_CHRONIC_PE_FACTOR: "
        f"{summary['acute_and_chronic_factor']:.3f}"
    )
    print(
        f"train series before: {summary['before']['total']}, "
        f"after_undersampling: {summary['after_undersampling']['total']}, "
        f"after_resampling: {summary['after']['total']}"
    )
    print(
        f"chronic positive before: {summary['before']['chronic_positive']}, "
        f"after_undersampling: {summary['after_undersampling']['chronic_positive']}, "
        f"after_resampling: {summary['after']['chronic_positive']}"
    )
    print(
        "acute_and_chronic positive before: "
        f"{summary['before']['acute_and_chronic_positive']}, "
        f"after_undersampling: {summary['after_undersampling']['acute_and_chronic_positive']}, "
        f"after_resampling: {summary['after']['acute_and_chronic_positive']}"
    )
    print(
        f"chronic_only before: {summary['before']['chronic_only']}, "
        f"after_undersampling: {summary['after_undersampling']['chronic_only']}, "
        f"after_resampling: {summary['after']['chronic_only']}"
    )
    print(
        f"other_series before: {summary['before']['other_series']}, "
        f"after_undersampling: {summary['after_undersampling']['other_series']}, "
        f"after_resampling: {summary['after']['other_series']}"
    )
    print("=" * 60)
