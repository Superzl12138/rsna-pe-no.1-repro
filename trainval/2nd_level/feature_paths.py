import os
from pathlib import Path


def resolve_feature_paths(backbone_name):
    script_dir = Path(__file__).resolve().parent
    default_feature_dir = script_dir.parent / backbone_name / "features0"
    env_key = f"{backbone_name.upper()}_FEATURE_DIR"
    feature_dir = Path(os.environ.get(env_key, os.environ.get("SECOND_LEVEL_FEATURE_DIR", str(default_feature_dir))))
    train_path = feature_dir / "feature_train.npy"
    valid_path = feature_dir / "feature_valid.npy"

    print("=" * 60)
    print("2nd-level Feature Input")
    print("=" * 60)
    print(f"backbone: {backbone_name}")
    print(f"{env_key}: {os.environ.get(env_key, '')}")
    print(f"SECOND_LEVEL_FEATURE_DIR: {os.environ.get('SECOND_LEVEL_FEATURE_DIR', '')}")
    print(f"resolved_feature_dir: {feature_dir}")
    print(f"feature_train_path: {train_path}")
    print(f"feature_valid_path: {valid_path}")
    print("=" * 60)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing feature_train.npy: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing feature_valid.npy: {valid_path}")

    return train_path, valid_path
