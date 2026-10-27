#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import pickle
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# Global default paths (relative to script directory)
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "processed_data", "disease_data")
DATA_ROOT = os.environ.get("DDO_DATA_ROOT", DEFAULT_DATA_ROOT)

DEFAULT_RAW_PATHS = {
    "DXY": os.path.join(SCRIPT_DIR, "raw_data", "DXY"),
    "GMD": os.path.join(SCRIPT_DIR, "raw_data", "GMD"),
    "CMD": os.path.join(SCRIPT_DIR, "raw_data", "CMD"),
}

# Pre-split directory (train.json / dev.json / test.json)
CMD_DATA_DIR = os.path.join(SCRIPT_DIR, "raw_data", "CMD")


# ============================================================
# Utility functions
# ============================================================

def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_original_dataset_file(path: str):
    """
    Auto-detect and load raw data file.

    Supports: .json, .pk, .pkl, .pickle, and files with pickle magic bytes.
    """
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")

    with open(path, "rb") as f:
        magic = f.read(2)

    lower_path = path.lower()
    is_pickle = (
        lower_path.endswith(".pk")
        or lower_path.endswith(".pkl")
        or lower_path.endswith(".pickle")
        or magic.startswith(b"\x80")
    )

    if is_pickle:
        with open(path, "rb") as f:
            return pickle.load(f)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_dir(dataset_name: str):
    return os.path.join(DATA_ROOT, dataset_name)


def get_turn_data_dir():
    return os.path.join(DATA_ROOT, "turn_data")


# ============================================================
# Read and save base data
# ============================================================

def load_dataset(dataset_name: str, stage: str):
    path = os.path.join(get_dataset_dir(dataset_name), f"{stage}.json")
    with open(path, "r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def unify_disease_label(disease_label: str):
    if disease_label in ["小儿手足口病"]:
        return "手足口病"
    if disease_label in ["哮喘"]:
        return "支气管哮喘"
    return disease_label


def convert_data_format(dataset_name: str, original_dataset: dict, stage: str):
    if stage not in original_dataset:
        return []

    original_stage_dataset = original_dataset[stage]
    converted_dataset = []

    for sample_id, data in enumerate(original_stage_dataset, start=1):
        disease_label = data["disease_tag"]

        if dataset_name == "CMD":
            disease_label = disease_label.split("@")[1]

        disease_label = unify_disease_label(disease_label)

        if dataset_name == "DXY":
            self_report = data["goal"]["explicit_inform_slots"]
            implicit_symptoms = data["goal"]["implicit_inform_slots"]
        else:
            self_report = data["explicit_inform_slots"]
            implicit_symptoms = data["implicit_inform_slots"]

        if len(self_report) == 0:
            continue

        all_symptoms = {**self_report, **implicit_symptoms}

        converted_dataset.append({
            "id": sample_id,
            "disease_label": disease_label,
            "self_report": self_report,
            "all_symptoms": all_symptoms,
        })

    return converted_dataset


def save_preprocessed_dataset(dataset_name: str, dataset: List[dict], stage: str):
    path = os.path.join(get_dataset_dir(dataset_name), f"{stage}.json")
    save_json(dataset, path)


def calculate_symptom_frequencies(dataset_name: str, dataset: List[dict]):
    """
    Calculate positive symptom frequency per disease as empirical_knowledge.
    Only counts status=True symptoms.
    """
    symptom_frequency = {}
    disease_count = {}

    for data in dataset:
        disease_label = data["disease_label"]
        all_symptoms = data["all_symptoms"]

        if disease_label not in symptom_frequency:
            symptom_frequency[disease_label] = {}
            disease_count[disease_label] = 0

        disease_count[disease_label] += 1

        for symptom, status in all_symptoms.items():
            if status:
                symptom_frequency[disease_label][symptom] = symptom_frequency[disease_label].get(symptom, 0) + 1

    for disease_label, symptoms in symptom_frequency.items():
        total_cases = disease_count[disease_label]

        symptom_frequency[disease_label] = dict(
            sorted(
                {
                    symptom: round(frequency / total_cases, 3)
                    for symptom, frequency in symptoms.items()
                    if round(frequency / total_cases, 3) >= 0.02
                }.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        )

    path = os.path.join(get_dataset_dir(dataset_name), "empirical_knowledge.json")
    save_json(symptom_frequency, path)


def save_disease_symptom_corpurs(
    dataset_name: str,
    train_dataset: List[dict],
    dev_dataset: Optional[List[dict]],
    test_dataset: List[dict],
):
    """
    Save disease set and symptom set.
    Note: filename 'corpurs' kept for backward compatibility.
    """
    disease_list = []
    symptom_list = []

    for dataset in [train_dataset, dev_dataset or [], test_dataset]:
        for data in dataset:
            if data["disease_label"] not in disease_list:
                disease_list.append(data["disease_label"])

            for symptom in data["all_symptoms"].keys():
                if symptom not in symptom_list:
                    symptom_list.append(symptom)

    dataset_dir = get_dataset_dir(dataset_name)
    ensure_dir(dataset_dir)

    disease_path = os.path.join(dataset_dir, "disease_corpurs.txt")
    symptom_path = os.path.join(dataset_dir, "symptom_corpurs.txt")

    with open(disease_path, "w", encoding="utf-8") as disease_corpurs_file:
        for disease in disease_list:
            disease_corpurs_file.write(disease + "\n")

    with open(symptom_path, "w", encoding="utf-8") as symptom_corpurs_file:
        for symptom in symptom_list:
            symptom_corpurs_file.write(symptom + "\n")


def get_disease_index_dict(dataset_name: str):
    disease_index_dict = {}
    path = os.path.join(get_dataset_dir(dataset_name), "disease_corpurs.txt")

    with open(path, "r", encoding="utf-8") as disease_corpurs_file:
        lines = disease_corpurs_file.readlines()

    for line in lines:
        disease = line.strip()
        if disease:
            disease_index_dict[disease] = len(disease_index_dict)

    return disease_index_dict


def get_symptom_index_dict(dataset_name: str):
    symptom_index_dict = {}
    path = os.path.join(get_dataset_dir(dataset_name), "symptom_corpurs.txt")

    with open(path, "r", encoding="utf-8") as symptom_corpurs_file:
        lines = symptom_corpurs_file.readlines()

    for line in lines:
        symptom = line.strip()
        if symptom:
            symptom_index_dict[symptom] = len(symptom_index_dict)

    return symptom_index_dict


# ============================================================
# Stage 1: raw data -> train/dev/test/knowledge files
# ============================================================

def _load_presplit_dataset(data_dir: str, stage: str):
    """Load data from pre-split GMD-format directory."""
    path = os.path.join(data_dir, f"{stage}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_dataset(dataset_name: str, seed: int = 42, raw_paths: Optional[Dict[str, str]] = None):
    if raw_paths is None:
        raw_paths = DEFAULT_RAW_PATHS

    original_dataset_filepath = os.path.expanduser(raw_paths[dataset_name])

    # ---- GMD: data is pre-split, load train.json / dev.json / test.json directly ----
    if os.path.isdir(original_dataset_filepath):
        print(f"[{dataset_name}] loading pre-split data from directory: {original_dataset_filepath}")
        train_dataset = _load_presplit_dataset(original_dataset_filepath, "train")
        dev_dataset = _load_presplit_dataset(original_dataset_filepath, "dev")
        test_dataset = _load_presplit_dataset(original_dataset_filepath, "test")

        if not train_dataset:
            raise ValueError(f"[{dataset_name}] train.json is empty or missing.")
        if not test_dataset:
            raise ValueError(f"[{dataset_name}] test.json is empty or missing.")

        rng = random.Random(seed)
        rng.shuffle(train_dataset)
    else:
        # ---- Original flow: read from pickle/json file and convert ----
        original_dataset = load_original_dataset_file(original_dataset_filepath)

        if not isinstance(original_dataset, dict):
            raise TypeError(f"[{dataset_name}] Raw data should be dict, got {type(original_dataset)}")

        print(f"[{dataset_name}] raw path: {original_dataset_filepath}")
        print(f"[{dataset_name}] original keys: {list(original_dataset.keys())}")

        train_dataset = convert_data_format(dataset_name, original_dataset, "train")
        dev_dataset = convert_data_format(dataset_name, original_dataset, "dev")
        test_dataset = convert_data_format(dataset_name, original_dataset, "test")

        if not train_dataset:
            raise ValueError(f"[{dataset_name}] train dataset is empty.")
        if not test_dataset:
            raise ValueError(f"[{dataset_name}] test dataset is empty.")

        dev_ratio = {"DXY": 1, "GMD": 1, "CMD": 0.5}

        if not dev_dataset:
            train_dataset, dev_dataset = train_test_split(
                train_dataset,
                test_size=(len(test_dataset) / len(train_dataset)) * dev_ratio[dataset_name],
                stratify=[data["disease_label"] for data in train_dataset],
                random_state=seed,
            )
        else:
            rng = random.Random(seed)
            rng.shuffle(train_dataset)

    save_disease_symptom_corpurs(
        dataset_name=dataset_name,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
    )

    save_preprocessed_dataset(dataset_name, train_dataset, "train")
    save_preprocessed_dataset(dataset_name, dev_dataset, "dev")
    save_preprocessed_dataset(dataset_name, test_dataset, "test")

    calculate_symptom_frequencies(dataset_name, train_dataset)

    print(f"[{dataset_name}] saved to {get_dataset_dir(dataset_name)}")
    print(f"[{dataset_name}] train={len(train_dataset)}, dev={len(dev_dataset)}, test={len(test_dataset)}")


# ============================================================
# Stage 2: turn data augmentation functions
# ============================================================

def sample_combinations(
    num_items: int,
    depth: int,
    max_comb_per_depth: Optional[int],
    rng: random.Random,
):
    """
    Select 'depth' items from 'num_items' implicit symptoms.

    max_comb_per_depth=None means keep all combinations.
    max_comb_per_depth=K caps random samples per depth to avoid explosion.
    """
    if depth < 0 or depth > num_items:
        return []

    total_comb = math.comb(num_items, depth)

    if max_comb_per_depth is None or max_comb_per_depth >= total_comb:
        return list(combinations(range(num_items), depth))

    sampled = set()
    while len(sampled) < max_comb_per_depth:
        combo = tuple(sorted(rng.sample(range(num_items), depth)))
        sampled.add(combo)

    return list(sampled)


def make_probe_state_key(symptoms: Dict[str, bool]) -> str:
    """
    Create a stable string key for the current turn's symptom state.
    Example: cough=True|fever=True|vomit=False
    """
    return "|".join(
        f"{symptom}={int(status)}"
        for symptom, status in sorted(symptoms.items(), key=lambda x: x[0])
    )


def make_turn_item(
    dataset_name: str,
    sample_id: int,
    turn_id: int,
    candidate_diseases: List[str],
    last_turn_symptoms: Dict[str, bool],
    current_turn_symptoms: Dict[str, bool],
    disease_label: str,
    aug_depth: int,
    aug_repeat_id: int,
    added_symptoms: List[str],
    branch_enabled: bool,
    repeat_enabled: bool,
):
    probe_group_id = f"{dataset_name}-{sample_id}"
    probe_state_key = make_probe_state_key(current_turn_symptoms)

    probe_uid = (
        f"{probe_group_id}"
        f"|turn={turn_id}"
        f"|depth={aug_depth}"
        f"|repeat={aug_repeat_id}"
        f"|branch={int(branch_enabled)}"
        f"|repeat_on={int(repeat_enabled)}"
        f"|state={probe_state_key}"
    )

    return {
        "dataset_name": dataset_name,
        "id": sample_id,
        "turn_id": turn_id,
        "probe_group_id": probe_group_id,
        "probe_uid": probe_uid,
        "probe_state_key": probe_state_key,
        "probe_label": disease_label,
        "candidate_diseases": candidate_diseases,
        "aug_depth": aug_depth,
        "aug_repeat_id": aug_repeat_id,
        "added_symptoms": added_symptoms,
        "branch_enabled": branch_enabled,
        "repeat_enabled": repeat_enabled,
        "last_turn_True": [s for s, st in last_turn_symptoms.items() if st],
        "last_turn_False": [s for s, st in last_turn_symptoms.items() if not st],
        "current_turn_True": [s for s, st in current_turn_symptoms.items() if st],
        "current_turn_False": [s for s, st in current_turn_symptoms.items() if not st],
        "disease_label": disease_label,
    }

def get_repeat_times(
    depth: int,
    use_repeat: bool,
    repeat_base: int,
    max_repeat_n: Optional[int],
    max_depth: Optional[int] = None,
    descending: bool = False,
):
    """
    depth definition:
    - depth=0: self_report only, e.g. ab.
    - depth=1: add 1 implicit symptom, e.g. abc.
    - depth=2: add 2 implicit symptoms, e.g. abcd.

    use_repeat=False: always repeat 1 time.
    use_repeat=True and descending=False: default repeats = depth + repeat_base (ascending).

    include_initial_state=True, repeat_base=1:
        ab x1, abc x2, abcd x3

    include_initial_state=False, repeat_base=0:
        abc x1, abcd x2, abcde x3

    descending=True: repeats decrease from max_depth to 1.
        depth=0 -> max_depth, depth=1 -> max_depth-1, ..., depth=max_depth -> 1
    """
    if not use_repeat:
        return 1

    if descending and max_depth is not None:
        repeat_times = max(1, max_depth - depth + 1)
    else:
        repeat_times = max(1, depth + repeat_base)

    if max_repeat_n is not None:
        repeat_times = min(repeat_times, max_repeat_n)

    return repeat_times


def generate_single_path_turn_items(
    dataset_name: str,
    sample_id: int,
    self_report: Dict[str, bool],
    implicit_items: List[Tuple[str, bool]],
    candidate_diseases: List[str],
    disease_label: str,
    use_repeat: bool,
    max_repeat_n: Optional[int],
    repeat_base: int,
    repeat_descending: bool = False,
    max_aug_depth: Optional[int] = None,
):
    """
    DDO original single-path cumulative expansion.

    self_report = ab, implicit = c,d,e
    Generates:
        abc
        abcd
        abcde

    use_repeat=True:
        abc x1
        abcd x2
        abcde x3

    repeat_descending=True:
        abc x3
        abcd x2
        abcde x1

    max_aug_depth truncates implicit symptom count to control max depth.
    """
    if max_aug_depth is not None:
        implicit_items = implicit_items[:max_aug_depth]

    turn_datas = []
    last_turn_symptoms = self_report.copy()
    max_depth = len(implicit_items)

    for depth, (symptom, existence) in enumerate(implicit_items, start=1):
        current_turn_symptoms = {**last_turn_symptoms, symptom: existence}

        repeat_times = get_repeat_times(
            depth=depth,
            use_repeat=use_repeat,
            repeat_base=repeat_base,
            max_repeat_n=max_repeat_n,
            max_depth=max_depth,
            descending=repeat_descending,
        )

        for repeat_id in range(1, repeat_times + 1):
            item = make_turn_item(
                dataset_name=dataset_name,
                sample_id=sample_id,
                turn_id=depth,
                candidate_diseases=candidate_diseases,
                last_turn_symptoms=last_turn_symptoms,
                current_turn_symptoms=current_turn_symptoms,
                disease_label=disease_label,
                aug_depth=depth,
                aug_repeat_id=repeat_id,
                added_symptoms=[symptom],
                branch_enabled=False,
                repeat_enabled=use_repeat,
            )
            turn_datas.append(item)

        last_turn_symptoms = current_turn_symptoms.copy()

    return turn_datas


def generate_branch_turn_items(
    dataset_name: str,
    sample_id: int,
    self_report: Dict[str, bool],
    implicit_items: List[Tuple[str, bool]],
    candidate_diseases: List[str],
    disease_label: str,
    use_repeat: bool,
    include_initial_state: bool,
    max_aug_depth: Optional[int],
    max_comb_per_depth: Optional[int],
    max_repeat_n: Optional[int],
    repeat_base: int,
    rng: random.Random,
    repeat_descending: bool = False,
    repeat_order: int = 1,
    seed: int = 42,
):
    """
    Branch combination expansion.

    self_report = ab, implicit = c,d,e

    include_initial_state=True:
        depth=0: ab
        depth=1: abc, abd, abe
        depth=2: abcd, abce, abde
        depth=3: abcde

    use_repeat=True and repeat_base=1:
        ab x1
        abc/abd/abe each x2
        abcd/abce/abde each x3
        abcde x4
    """
    turn_datas = []
    implicit_num = len(implicit_items)
    real_max_depth = implicit_num if max_aug_depth is None else min(max_aug_depth, implicit_num)
    branch_max_depth = real_max_depth

    if implicit_num == 0:
        if include_initial_state:
            repeat_times = get_repeat_times(
                depth=0,
                use_repeat=use_repeat,
                repeat_base=repeat_base,
                max_repeat_n=max_repeat_n,
                max_depth=branch_max_depth,
                descending=repeat_descending,
            )

            for repeat_id in range(1, repeat_times + 1):
                item = make_turn_item(
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    turn_id=0,
                    candidate_diseases=candidate_diseases,
                    last_turn_symptoms=self_report,
                    current_turn_symptoms=self_report,
                    disease_label=disease_label,
                    aug_depth=0,
                    aug_repeat_id=repeat_id,
                    added_symptoms=[],
                    branch_enabled=True,
                    repeat_enabled=use_repeat,
                )
                turn_datas.append(item)

        return turn_datas

    start_depth = 0 if include_initial_state else 1

    for depth in range(start_depth, real_max_depth + 1):
        if depth == 0:
            combos = [tuple()]
        else:
            combos = sample_combinations(
                num_items=implicit_num,
                depth=depth,
                max_comb_per_depth=max_comb_per_depth,
                rng=rng,
            )

        repeat_times = get_repeat_times(
            depth=depth,
            use_repeat=use_repeat,
            repeat_base=repeat_base,
            max_repeat_n=max_repeat_n,
            max_depth=branch_max_depth,
            descending=repeat_descending,
        )

        for combo in combos:
            selected_items = [implicit_items[index] for index in combo]

            # When repeat_order>1, generate multiple permutations of symptoms within combo,
            # each producing a different last_turn -> current_turn transition path
            if depth <= 1 or repeat_order <= 1:
                perms = [selected_items]
            else:
                case_rng = random.Random(hash(sample_id) ^ seed ^ hash(combo))
                seen = {tuple(selected_items)}
                perms = [selected_items]
                for _ in range(1, repeat_order):
                    perm = list(selected_items)
                    case_rng.shuffle(perm)
                    t = tuple(perm)
                    if t not in seen:
                        seen.add(t)
                        perms.append(perm)

            for perm in perms:
                selected_symptom_names = [symptom for symptom, _ in perm]

                current_turn_symptoms = self_report.copy()
                for symptom, existence in perm:
                    current_turn_symptoms[symptom] = existence

                last_turn_symptoms = self_report.copy()
                if depth > 0:
                    for symptom, existence in perm[:-1]:
                        last_turn_symptoms[symptom] = existence

                for repeat_id in range(1, repeat_times + 1):
                    item = make_turn_item(
                        dataset_name=dataset_name,
                        sample_id=sample_id,
                        turn_id=depth,
                        candidate_diseases=candidate_diseases,
                        last_turn_symptoms=last_turn_symptoms,
                        current_turn_symptoms=current_turn_symptoms,
                        disease_label=disease_label,
                        aug_depth=depth,
                        aug_repeat_id=repeat_id,
                        added_symptoms=selected_symptom_names,
                        branch_enabled=True,
                        repeat_enabled=use_repeat,
                    )
                    turn_datas.append(item)

    return turn_datas


# ============================================================
# Stage 2: generate turn data main function
# ============================================================

def preprocess_turn_dataset(
    dataset_name: str,
    seed: int = 42,
    use_branch: bool = False,
    use_repeat: bool = False,
    include_initial_state: bool = False,
    max_aug_depth: Optional[int] = None,
    max_comb_per_depth: Optional[int] = 20,
    max_repeat_n: Optional[int] = None,
    repeat_descending: bool = False,
    repeat_order: int = 1,
):
    """Generate DDO training turn data."""
    rng = random.Random(seed)

    all_diseases = list(get_disease_index_dict(dataset_name).keys())

    dataset = load_dataset(dataset_name=dataset_name, stage="train")
    turn_datas = []

    repeat_base = 1 if include_initial_state else 0

    for data in tqdm(dataset, desc=f"Building turn data for {dataset_name}"):
        sample_id = data["id"]
        disease_label = data["disease_label"]
        self_report = data["self_report"].copy()

        # Candidate diseases: real disease first + 4 random others, shuffle
        others = [d for d in all_diseases if d != disease_label]
        rng.shuffle(others)
        candidate_diseases = [disease_label] + others[:11]
        rng.shuffle(candidate_diseases)

        # Implicit symptoms = all_symptoms - self_report
        new_symptoms = {
            symptom: existence
            for symptom, existence in data["all_symptoms"].items()
            if symptom not in self_report
        }

        implicit_items = list(new_symptoms.items())

        if use_branch:
            items = generate_branch_turn_items(
                dataset_name=dataset_name,
                sample_id=sample_id,
                self_report=self_report,
                implicit_items=implicit_items,
                candidate_diseases=candidate_diseases,
                disease_label=disease_label,
                use_repeat=use_repeat,
                include_initial_state=include_initial_state,
                max_aug_depth=max_aug_depth,
                max_comb_per_depth=max_comb_per_depth,
                max_repeat_n=max_repeat_n,
                repeat_base=repeat_base,
                rng=rng,
                repeat_descending=repeat_descending,
                repeat_order=repeat_order,
                seed=seed,
            )
        else:
            items = []
            if repeat_order <= 1 or len(implicit_items) <= 1:
                items = generate_single_path_turn_items(
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    self_report=self_report,
                    implicit_items=implicit_items,
                    candidate_diseases=candidate_diseases,
                    disease_label=disease_label,
                    use_repeat=use_repeat,
                    max_repeat_n=max_repeat_n,
                    repeat_base=repeat_base,
                    repeat_descending=repeat_descending,
                    max_aug_depth=max_aug_depth,
                )
            else:
                # Original order
                items.extend(generate_single_path_turn_items(
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    self_report=self_report,
                    implicit_items=implicit_items,
                    candidate_diseases=candidate_diseases,
                    disease_label=disease_label,
                    use_repeat=use_repeat,
                    max_repeat_n=max_repeat_n,
                    repeat_base=repeat_base,
                    repeat_descending=repeat_descending,
                    max_aug_depth=max_aug_depth,
                ))
                # Shuffle order, generate full trajectory per order
                case_rng = random.Random(hash(sample_id) ^ seed)
                seen = {tuple(implicit_items)}
                for _ in range(1, repeat_order):
                    shuffled = list(implicit_items)
                    case_rng.shuffle(shuffled)
                    if tuple(shuffled) in seen:
                        continue
                    seen.add(tuple(shuffled))
                    items.extend(generate_single_path_turn_items(
                        dataset_name=dataset_name,
                        sample_id=sample_id,
                        self_report=self_report,
                        implicit_items=shuffled,
                        candidate_diseases=candidate_diseases,
                        disease_label=disease_label,
                        use_repeat=use_repeat,
                        max_repeat_n=max_repeat_n,
                        repeat_base=repeat_base,
                        repeat_descending=repeat_descending,
                        max_aug_depth=max_aug_depth,
                    ))

        turn_datas.extend(items)

    # Global shuffle to avoid same-order sample clustering
    rng.shuffle(turn_datas)

    save_path = os.path.join(get_turn_data_dir(), f"{dataset_name}_train_turn.json")
    save_json(turn_datas, save_path)

    print(f"[{dataset_name}] turn data saved to {save_path}")
    print(f"[{dataset_name}] total turn samples: {len(turn_datas)}")
    print(f"[{dataset_name}] use_branch={use_branch}, use_repeat={use_repeat}, include_initial_state={include_initial_state}, repeat_descending={repeat_descending}, repeat_order={repeat_order}")

    return turn_datas


def load_turn_datasets(dataset_name: str):
    """
    Reads DATA_ROOT/turn_data/{dataset_name}_train_turn.json by default.
    """
    path = os.path.join(get_turn_data_dir(), f"{dataset_name}_train_turn.json")

    with open(path, "r", encoding="utf-8") as data_file:
        train_dataset = json.load(data_file)

    return train_dataset


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DXY", "GMD", "CMD"],
        choices=["DXY", "GMD", "CMD"],
        help="Datasets to process.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for processed and augmented turn data.",
    )

    parser.add_argument(
        "--dxy_raw_path",
        type=str,
        default=DEFAULT_RAW_PATHS["DXY"],
        help="DXY raw data path, supports .json/.pk/.pkl/.pickle.",
    )

    parser.add_argument(
        "--gmd_raw_path",
        type=str,
        default=DEFAULT_RAW_PATHS["GMD"],
        help="GMD raw data path, supports .json/.pk/.pkl/.pickle.",
    )

    parser.add_argument(
        "--cmd_raw_path",
        type=str,
        default=DEFAULT_RAW_PATHS["CMD"],
        help="CMD raw data path, supports .json/.pk/.pkl/.pickle.",
    )

    parser.add_argument(
        "--do_preprocess_dataset",
        type=str2bool,
        default=True,
        help="Generate train/dev/test and empirical_knowledge from raw data.",
    )

    parser.add_argument(
        "--do_preprocess_turn",
        type=str2bool,
        default=True,
        help="Generate DDO training turn data.",
    )

    parser.add_argument(
        "--use_branch",
        type=str2bool,
        default=False,
        help="Use branch combination augmentation. False = DDO original single-path expansion.",
    )

    parser.add_argument(
        "--use_repeat",
        type=str2bool,
        default=False,
        help="Use depth-based repeat augmentation. False = each state appears once.",
    )

    parser.add_argument(
        "--include_initial_state",
        type=str2bool,
        default=False,
        help="Include self_report initial state (e.g. ab -> disease). Usually True with branch mode.",
    )

    parser.add_argument(
        "--max_aug_depth",
        type=int,
        default=None,
        help="Max branch depth. None = all implicit symptoms. Only effective when use_branch=True.",
    )

    parser.add_argument(
        "--max_comb_per_depth",
        type=int,
        default=-1,
        help="Max combinations per depth. -1 = keep all. Only effective when use_branch=True.",
    )

    parser.add_argument(
        "--max_repeat_n",
        type=int,
        default=None,
        help="Max repeats. None = unlimited. Only effective when use_repeat=True.",
    )

    parser.add_argument(
        "--repeat_descending",
        type=str2bool,
        default=False,
        help="Decreasing repeats? False = ascending (1->n), True = descending (n->1). Only effective when use_repeat=True.",
    )

    parser.add_argument(
        "--repeat_order",
        type=int,
        default=1,
        help="Num permutations of implicit symptom order in single-path mode. 1 = original only. Only when use_branch=False.",
    )

    return parser.parse_args()


def main():
    global DATA_ROOT

    args = parse_args()
    DATA_ROOT = os.path.expanduser(args.data_root)

    raw_paths = {
        "DXY": args.dxy_raw_path,
        "GMD": args.gmd_raw_path,
        "CMD": args.cmd_raw_path,
    }

    if args.max_comb_per_depth == -1:
        max_comb_per_depth = None
    else:
        max_comb_per_depth = args.max_comb_per_depth

    print(f"DATA_ROOT = {DATA_ROOT}")

    for dataset_name in args.datasets:
        if args.do_preprocess_dataset:
            preprocess_dataset(
                dataset_name=dataset_name,
                seed=args.seed,
                raw_paths=raw_paths,
            )

        if args.do_preprocess_turn:
            preprocess_turn_dataset(
                dataset_name=dataset_name,
                seed=args.seed,
                use_branch=args.use_branch,
                use_repeat=args.use_repeat,
                include_initial_state=args.include_initial_state,
                max_aug_depth=args.max_aug_depth,
                max_comb_per_depth=max_comb_per_depth,
                max_repeat_n=args.max_repeat_n,
                repeat_descending=args.repeat_descending,
                repeat_order=args.repeat_order,
            )


if __name__ == "__main__":
    main()
