import os
import pickle


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(ROOT_DIR, "trainval", "process_input", "split2")

LIMIT_CONFIG = {
    "series_list_train.pickle": "TRAIN_SERIES_LIMIT",
    "series_list_valid.pickle": "VALID_SERIES_LIMIT",
    "image_list_train.pickle": None,
    "image_list_valid.pickle": None,
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path, data):
    if os.path.islink(path):
        os.unlink(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def backup_path(name):
    return os.path.join(SPLIT_DIR, f".full_{name}")


def load_full_pickle(name):
    original_path = os.path.join(SPLIT_DIR, name)
    full_path = backup_path(name)
    if os.path.exists(full_path):
        return load_pickle(full_path)
    data = load_pickle(original_path)
    with open(full_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def build_image_list(series_list, series_dict):
    image_list = []
    for series_id in series_list:
        image_list += list(series_dict[series_id]["sorted_image_list"])
    return image_list


def parse_limit(name):
    env_name = LIMIT_CONFIG[name]
    if env_name is None:
        return None
    raw = os.environ.get(env_name, "").strip()
    if raw == "":
        return None
    value = int(raw)
    if value <= 0:
        return None
    return value


def main():
    series_dict = load_pickle(os.path.join(SPLIT_DIR, "series_dict.pickle"))

    full_series_train = load_full_pickle("series_list_train.pickle")
    full_series_valid = load_full_pickle("series_list_valid.pickle")
    load_full_pickle("image_list_train.pickle")
    load_full_pickle("image_list_valid.pickle")

    train_limit = parse_limit("series_list_train.pickle")
    valid_limit = parse_limit("series_list_valid.pickle")

    series_list_train = full_series_train[:train_limit] if train_limit is not None else full_series_train
    series_list_valid = full_series_valid[:valid_limit] if valid_limit is not None else full_series_valid

    image_list_train = build_image_list(series_list_train, series_dict)
    image_list_valid = build_image_list(series_list_valid, series_dict)

    dump_pickle(os.path.join(SPLIT_DIR, "series_list_train.pickle"), series_list_train)
    dump_pickle(os.path.join(SPLIT_DIR, "series_list_valid.pickle"), series_list_valid)
    dump_pickle(os.path.join(SPLIT_DIR, "image_list_train.pickle"), image_list_train)
    dump_pickle(os.path.join(SPLIT_DIR, "image_list_valid.pickle"), image_list_valid)

    print(
        f"trainval subset ready: train_series={len(series_list_train)}, "
        f"valid_series={len(series_list_valid)}, "
        f"train_images={len(image_list_train)}, valid_images={len(image_list_valid)}"
    )


if __name__ == "__main__":
    main()
