import os

import numpy as np


def _parse_env_factor(env_key, default=1.0):
    value = os.environ.get(env_key, str(default))
    try:
        return max(1.0, float(value))
    except ValueError:
        raise ValueError(f"Invalid float value for {env_key}: {value}")


def _summarize_series_list(series_list, series_dict):
    summary = {
        "total": len(series_list),
        "chronic_positive": 0,
        "acute_and_chronic_positive": 0,
        "chronic_only": 0,
        "other_series": 0,
    }
    for series_id in series_list:
        is_chronic = int(series_dict[series_id]["chronic_pe"]) == 1
        is_acute_and_chronic = int(series_dict[series_id]["acute_and_chronic_pe"]) == 1
        if is_chronic:
            summary["chronic_positive"] += 1
        if is_acute_and_chronic:
            summary["acute_and_chronic_positive"] += 1
        if is_acute_and_chronic:
            continue
        if is_chronic:
            summary["chronic_only"] += 1
        else:
            summary["other_series"] += 1
    return summary


def build_oversampled_series_list(series_list, series_dict, seed):
    enabled = os.environ.get("ENABLE_2ND_LEVEL_OVERSAMPLING", "0") == "1"
    chronic_factor = _parse_env_factor("OVERSAMPLE_CHRONIC_PE_FACTOR", default=1.0)
    acute_and_chronic_factor = _parse_env_factor(
        "OVERSAMPLE_ACUTE_AND_CHRONIC_PE_FACTOR",
        default=1.0,
    )

    original_summary = _summarize_series_list(series_list, series_dict)
    if not enabled:
        return list(series_list), {
            "enabled": False,
            "chronic_factor": chronic_factor,
            "acute_and_chronic_factor": acute_and_chronic_factor,
            "before": original_summary,
            "after": original_summary.copy(),
        }

    rng = np.random.RandomState(seed)
    expanded_series_list = []
    for series_id in series_list:
        is_chronic = int(series_dict[series_id]["chronic_pe"]) == 1
        is_acute_and_chronic = int(series_dict[series_id]["acute_and_chronic_pe"]) == 1
        target_factor = 1.0
        if is_chronic:
            target_factor = max(target_factor, chronic_factor)
        if is_acute_and_chronic:
            target_factor = max(target_factor, acute_and_chronic_factor)

        repeat_count = int(np.floor(target_factor))
        fractional = target_factor - repeat_count
        if rng.rand() < fractional:
            repeat_count += 1
        expanded_series_list.extend([series_id] * repeat_count)

    return expanded_series_list, {
        "enabled": True,
        "chronic_factor": chronic_factor,
        "acute_and_chronic_factor": acute_and_chronic_factor,
        "before": original_summary,
        "after": _summarize_series_list(expanded_series_list, series_dict),
    }


def print_oversampling_summary(summary):
    print("=" * 60)
    print("2nd-level Oversampling")
    print("=" * 60)
    print(f"enabled: {summary['enabled']}")
    print(f"OVERSAMPLE_CHRONIC_PE_FACTOR: {summary['chronic_factor']:.3f}")
    print(
        "OVERSAMPLE_ACUTE_AND_CHRONIC_PE_FACTOR: "
        f"{summary['acute_and_chronic_factor']:.3f}"
    )
    print(
        f"train series before: {summary['before']['total']}, "
        f"after: {summary['after']['total']}"
    )
    print(
        f"chronic positive before: {summary['before']['chronic_positive']}, "
        f"after: {summary['after']['chronic_positive']}"
    )
    print(
        "acute_and_chronic positive before: "
        f"{summary['before']['acute_and_chronic_positive']}, "
        f"after: {summary['after']['acute_and_chronic_positive']}"
    )
    print(
        f"chronic_only before: {summary['before']['chronic_only']}, "
        f"after: {summary['after']['chronic_only']}"
    )
    print(
        f"other_series before: {summary['before']['other_series']}, "
        f"after: {summary['after']['other_series']}"
    )
    print("=" * 60)
