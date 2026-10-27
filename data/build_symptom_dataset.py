
import os
import json
import pickle
import random
import argparse
from collections import OrderedDict, Counter
from itertools import combinations

import numpy as np
from tqdm import tqdm


# =========================================================
# Paths (relative to script directory)
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Arguments: keep the user's original switches, only add npy export switches.
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    default=os.path.join(SCRIPT_DIR, "raw_data", "CMD"),
    required=False,
    help="Input JSON/Pickle file path. Supports list, {'train': [...]}, {'train': [...], 'test': [...]}"
)

parser.add_argument(
    "--gmd_data_dir",
    type=str,
    default=os.path.join(SCRIPT_DIR, "raw_data", "GMD"),
    required=False,
    help="GMD-format data directory with train.json / dev.json / test.json. Takes priority over --input."
)

parser.add_argument(
    "--out_dir",
    type=str,
    default=os.path.join(SCRIPT_DIR, "processed_data", "diffusion_model_data"),
    required=False,
    help="Output directory"
)

parser.add_argument(
    "--test_size",
    type=int,
    default=50,
    help="If input has no test split, sample this many cases as test set (default 50)"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

parser.add_argument(
    "--use_dataset_split",
    type=int,
    default=1,
    help=(
        "If input is a dict with train/test/validate keys, use the given split (1) or re-split (0)."
    )
)

parser.add_argument(
    "--train_key",
    type=str,
    default="train",
    help="Train field name in dataset dict (default train)"
)

parser.add_argument(
    "--test_key",
    type=str,
    default="test",
    help="Test field name in dataset dict (default test)"
)

parser.add_argument(
    "--valid_key",
    type=str,
    default="validate",
    help="Validation field name in dataset dict (default validate)"
)

parser.add_argument(
    "--mode",
    type=str,
    default="repeated_permutation",
    choices=["single_order", "repeated_permutation", "all_subsets"],
    help=(
        "Data construction mode: "
        "single_order=original implicit symptom order; "
        "repeated_permutation=shuffle and repeat; "
        "all_subsets=all subsets (most data, may explode)"
    )
)

parser.add_argument(
    "--repeat_num",
    type=int,
    default=4,
    help="Number of random order permutations per case in repeated_permutation mode (default 4)"
)

parser.add_argument(
    "--deduplicate_states",
    type=int,
    default=0,
    help=(
        "Deduplicate identical current states (1=yes, 0=no). Keep 0 if using position repeat."
    )
)

parser.add_argument(
    "--max_implicit_for_all_subsets",
    type=int,
    default=6,
    help="Max implicit symptoms in all_subsets mode to prevent explosion (default 12)"
)

parser.add_argument(
    "--position_repeat",
    type=int,
    default=1,
    help="Enable position repeat augmentation (1=on, 0=off, default 1)"
)

parser.add_argument(
    "--position_repeat_base",
    type=int,
    default=0,
    help=(
        "Position repeat base n. If 0, auto-uses num_implicit m: step0 repeats m times, step1 m-1, etc."
    )
)

parser.add_argument(
    "--position_repeat_direction",
    type=str,
    default="ascending",
    choices=["descending", "ascending"],
    help=(
        "Position repeat direction: "
        "descending=step0 repeats n, step1 repeats n-1 (decreasing); "
        "ascending=step0 repeats 1, step1 repeats 2 (increasing)."
    )
)

parser.add_argument(
    "--target_shuffle_num",
    type=int,
    default=2,
    help=(
        "Number of internal orderings of positive target set per current state. 2 = original + 1 shuffle."
    )
)


# Rare implicit symptom targeted augmentation.
# Count lowest-frequency implicit symptoms in train_cases, then augment those rare symptoms in train samples.
parser.add_argument(
    "--rare_implicit_augment",
    type=int,
    default=1,
    help="Enable rare implicit symptom targeted augmentation (1=on, 0=off, default 1)"
)
parser.add_argument(
    "--rare_implicit_topk",
    type=int,
    default=6,
    help="How many lowest-frequency implicit symptoms to target for augmentation (default 10)"
)
parser.add_argument(
    "--apply_rare_aug_to_test",
    type=int,
    default=0,
    help="Apply rare augmentation to test split too (default 0, prevents distribution leak)"
)
parser.add_argument(
    "--rare_first_shuffle_num",
    type=int,
    default=2,
    help=(
        "Extra rare-first target orderings when a rare symptom stays in positive target. E.g. ab -> cde with rare d gives ab -> dce/dec. (default 2)"
    )
)
parser.add_argument(
    "--rare_split_shuffle_num",
    type=int,
    default=0,
    help=(
        "Number of target shuffles when a rare symptom is moved to current. E.g. ab -> cde with rare d gives abd -> ce/ec. (default 2)"
    )
)
parser.add_argument(
    "--rare_split_repeat_num",
    type=int,
    default=1,
    help="Extra repeats per shuffle version for rare-split samples (default 1)"
)
parser.add_argument(
    "--rare_split_anchor_mode",
    type=str,
    default="full_current",
    choices=["full_current", "pair_with_each_current"],
    help=(
        "Rare-split current construction: "
        "full_current=abd -> ce; "
        "pair_with_each_current=ad -> ce, bd -> ce (splits ab into pairs)."
    )
)

# DiffDiag matrix export related switches.
parser.add_argument(
    "--token_with_value",
    action="store_true",
    help="Default: symptom name only. Enable for symptom__T / symptom__F tokens with status."
)
parser.add_argument(
    "--include_false_implicit",
    type=int,
    default=1,
    help="Include False implicit symptoms in target (default 1, matches original behavior)"
)
parser.add_argument(
    "--include_false_explicit",
    type=int,
    default=1,
    help="Include False explicit symptoms in current state (default 1)"
)
parser.add_argument(
    "--vocab_from",
    type=str,
    default="train",
    choices=["train", "all"],
    help="Build vocab from train only (default) to prevent test leakage; 'all' avoids OOV in test."
)
parser.add_argument("--neg_weight", type=float, default=0.05)
parser.add_argument("--obs_weight", type=float, default=0.20)
parser.add_argument("--hidden_weight", type=float, default=1.00)
parser.add_argument("--other_pos_weight", type=float, default=0.50)
parser.add_argument("--item_emb_dim", type=int, default=64)

args = parser.parse_args()


# =========================================================
# Basic utilities
# =========================================================

def load_cases(input_path):
    """
    Supports:
    1) pickle/json: {'train': [...], 'test': [...], 'validate': [...]} -> use given split (when --use_dataset_split=1)
    2) pickle/json: {'train': [...]} -> split train/test using --test_size
    3) pickle/json: list -> split train/test using --test_size

    Added --use_dataset_split / --train_key / --test_key / --valid_key for compatibility
    with the original dataset reading logic.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".pkl", ".pickle", ".p"]:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # ---- Prefer dataset native train/test split ----
    if (
        bool(args.use_dataset_split)
        and isinstance(data, dict)
        and args.train_key in data
        and isinstance(data[args.train_key], list)
    ):
        train_cases = data.get(args.train_key, [])
        test_cases = data.get(args.test_key, [])

        if test_cases is None:
            test_cases = []

        if not isinstance(test_cases, list):
            raise ValueError(f"'{args.test_key}' field exists but is not a list.")

        if len(test_cases) == 0:
            train_cases, test_cases = split_cases(
                train_cases, test_size=args.test_size, seed=args.seed
            )
            return train_cases, test_cases, "split_from_train_field"

        return train_cases, test_cases, "input_train_test"

    # ---- Fallback: original logic ----
    if isinstance(data, dict):
        if "train" in data and isinstance(data["train"], list):
            train_cases = data["train"]
            test_cases = data.get("test", None)
            if isinstance(test_cases, list) and len(test_cases) > 0:
                return train_cases, test_cases, "input_train_test"
            train_cases, test_cases = split_cases(train_cases, test_size=args.test_size, seed=args.seed)
            return train_cases, test_cases, "split_from_train_field"
        raise ValueError("Top-level is dict but no list 'train' field found.")

    if isinstance(data, list):
        train_cases, test_cases = split_cases(data, test_size=args.test_size, seed=args.seed)
        return train_cases, test_cases, "split_from_list"

    raise ValueError("Unsupported input format; top-level must be dict or list.")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_cases(cases, test_size=50, seed=42):
    """Split by case to avoid leakage."""
    if test_size >= len(cases):
        raise ValueError(f"test_size={test_size} must be less than total cases {len(cases)}")

    rng = random.Random(seed)
    indices = list(range(len(cases)))
    test_indices = set(rng.sample(indices, test_size))

    train_cases, test_cases = [], []
    for i, case in enumerate(cases):
        if i in test_indices:
            test_cases.append(case)
        else:
            train_cases.append(case)

    return train_cases, test_cases


def normalize_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["true", "1", "yes", "y"]:
            return True
        if v in ["false", "0", "no", "n"]:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def slot_token(symptom, value):
    """Default matches your dataset.py: only symptom name. Optional value-aware mode is extra."""
    if args.token_with_value:
        return f"{symptom}__{'T' if normalize_value(value) else 'F'}"
    return str(symptom)


def to_ordered_dict(d, include_false=True):
    od = OrderedDict()
    if not isinstance(d, dict):
        return od
    for k, v in d.items():
        if normalize_value(v) is False and not include_false:
            continue
        od[k] = v
    return od


def build_token_vocab(cases):
    tokens = set()

    for case in cases:
        goal = case.get("goal", {})
        explicit = to_ordered_dict(goal.get("explicit_inform_slots", {}), include_false=bool(args.include_false_explicit))
        implicit = to_ordered_dict(goal.get("implicit_inform_slots", {}), include_false=bool(args.include_false_implicit))

        for s, v in explicit.items():
            tokens.add(slot_token(s, v))
        for s, v in implicit.items():
            tokens.add(slot_token(s, v))

    # Keep the same special ids as your script for JSONL compatibility.
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
    }

    for tok in sorted(tokens):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def tokens_to_ids(tokens, vocab):
    return [vocab.get(tok, vocab["[UNK]"]) for tok in tokens]


def state_key_from_tokens(tokens):
    return "||".join(sorted(tokens))


def compute_position_repeat_count(step, num_implicit):
    if not bool(args.position_repeat):
        return 1

    base = int(args.position_repeat_base)
    if base <= 0:
        base = int(num_implicit)

    if args.position_repeat_direction == "ascending":
        return max(int(step) + 1, 1)
    else:
        return max(base - int(step), 1)


def shuffled_ordered_dict(items, rng, shuffle_index):
    cur_items = list(items)
    if shuffle_index > 0 and len(cur_items) > 1:
        rng.shuffle(cur_items)
    return OrderedDict(cur_items)



# =========================================================
# Rare implicit symptom statistics and targeted augmentation
# =========================================================

def compute_rare_implicit_symptoms(cases, vocab, topk=10):
    """
    Count rare implicit symptoms only in train cases to avoid test leakage.

    Token unit matches model:
      - token_with_value=False: count symptom names;
      - token_with_value=True: count symptom__T / symptom__F.
    """
    counter = Counter()

    for case in cases:
        goal = case.get("goal", {})
        implicit = to_ordered_dict(
            goal.get("implicit_inform_slots", {}),
            include_false=bool(args.include_false_implicit)
        )
        for symptom, value in implicit.items():
            counter[slot_token(symptom, value)] += 1

    rare = []
    for token, count in sorted(counter.items(), key=lambda x: (x[1], x[0]))[:max(0, int(topk))]:
        rare.append({
            "token": token,
            "count": int(count),
            "token_id": int(vocab.get(token, vocab.get("[UNK]", 1)))
        })

    return rare


def sanitize_for_id(text):
    """Keep sample_id readable and stable even when symptom names contain spaces/slashes."""
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_").replace("|", "_")


def make_ordered_slots_from_items(items):
    od = OrderedDict()
    for s, v in items:
        od[s] = v
    return od


# =========================================================
# Sample construction: preserved from your dataset.py semantics
# =========================================================

def build_multi_positive_sample(
    case,
    current_slots,
    asked_implicit_slots,
    remaining_implicit_slots,
    vocab,
    sample_id,
    step,
    order_index=-1,
    target_shuffle_index=0,
    position_repeat_count=1,
    position_repeat_index=0,
    augment_type="base",
    rare_implicit_token="",
    rare_aug_index=-1,
):
    current_tokens = [slot_token(s, v) for s, v in current_slots.items()]
    positive_target_tokens = [slot_token(s, v) for s, v in remaining_implicit_slots.items()]
    positive_target_ids = tokens_to_ids(positive_target_tokens, vocab)

    legacy_target_token = positive_target_tokens[0] if positive_target_tokens else "[UNK]"
    legacy_target_id = positive_target_ids[0] if positive_target_ids else vocab["[UNK]"]

    sample = {
        "sample_id": sample_id,
        "consult_id": case.get("consult_id", ""),
        "disease_tag": case.get("disease_tag", ""),
        "self_repo": case.get("self_repo", ""),
        "dialog": case.get("dialog", []),
        "origin_inform_slots": case.get("goal", {}).get("origin_inform_slots", []),

        "mode": args.mode,
        "order_index": order_index,
        "step": step,

        "target_shuffle_index": target_shuffle_index,
        "position_repeat_count": position_repeat_count,
        "position_repeat_index": position_repeat_index,

        # Rare implicit symptom augmentation tag. "base" = original sample.
        "augment_type": augment_type,
        "rare_implicit_token": rare_implicit_token,
        "rare_aug_index": rare_aug_index,

        "current_slots": dict(current_slots),
        "asked_implicit_slots": dict(asked_implicit_slots),
        "remaining_implicit_slots": dict(remaining_implicit_slots),

        "current_tokens": current_tokens,
        "current_token_ids": tokens_to_ids(current_tokens, vocab),

        "positive_target_tokens": positive_target_tokens,
        "positive_target_ids": positive_target_ids,
        "num_positive_targets": len(positive_target_ids),

        "target_token": legacy_target_token,
        "target_token_id": legacy_target_id,
        "target_symptom": "",
        "target_value": "",
    }

    return sample


def build_rare_implicit_aug_samples(
    case,
    current_slots,
    asked_implicit_slots,
    remaining_items,
    vocab,
    sample_prefix,
    step,
    order_index,
    rng,
    repeat_count,
    rare_implicit_tokens,
):
    """
    Extra augmentation for rare implicit symptoms.

    Example: current=ab, positive target=cde, d is rare.
      1) rare_first: ab -> dce / dec (rare symptom first in target).
      2) rare_split: abd -> ce / ec (move rare to current, predict rest).

    With --rare_split_anchor_mode pair_with_each_current, rare_split becomes:
      ad -> ce, bd -> ce
    """
    if not bool(args.rare_implicit_augment) or not rare_implicit_tokens:
        return []

    augmented = []
    if not remaining_items:
        return augmented

    rare_positions = []
    for idx, (symptom, value) in enumerate(remaining_items):
        token = slot_token(symptom, value)
        if token in rare_implicit_tokens:
            rare_positions.append((idx, symptom, value, token))

    if not rare_positions:
        return augmented

    rare_aug_index = 0
    for rare_pos, rare_symptom, rare_value, rare_token in rare_positions:
        other_items = [item for idx, item in enumerate(remaining_items) if idx != rare_pos]
        rare_id_part = sanitize_for_id(rare_token)

        # 1) rare-first: put rare symptom first in target, shuffle the rest.
        # This repeat-weights multi-hot targets and reinforces rare symptoms for legacy target_token_id.
        # Skipped when rare_first_shuffle_num=0.
        rare_first_shuffle_num = int(args.rare_first_shuffle_num)
        if len(remaining_items) > 1 and rare_first_shuffle_num > 0:
            for shuffle_index in range(rare_first_shuffle_num):
                shuffled_others = list(other_items)
                if shuffle_index > 0 and len(shuffled_others) > 1:
                    rng.shuffle(shuffled_others)

                rare_first_items = [(rare_symptom, rare_value)] + shuffled_others
                rare_first_targets = make_ordered_slots_from_items(rare_first_items)

                for repeat_index in range(max(1, int(repeat_count))):
                    sample = build_multi_positive_sample(
                        case=case,
                        current_slots=current_slots,
                        asked_implicit_slots=asked_implicit_slots,
                        remaining_implicit_slots=rare_first_targets,
                        vocab=vocab,
                        sample_id=(
                            f"{sample_prefix}_rarefirst_r{rare_pos}_{rare_id_part}_"
                            f"shuf{shuffle_index}_rep{repeat_index}"
                        ),
                        step=step,
                        order_index=order_index,
                        target_shuffle_index=shuffle_index,
                        position_repeat_count=repeat_count,
                        position_repeat_index=repeat_index,
                        augment_type="rare_first",
                        rare_implicit_token=rare_token,
                        rare_aug_index=rare_aug_index,
                    )
                    augmented.append(sample)
                    rare_aug_index += 1

        # 2) rare-split: move rare symptom to current, predict remaining targets.
        # Skip if no remaining targets (samples_to_arrays drops empty targets).
        if len(other_items) > 0:
            split_current_variants = []

            if args.rare_split_anchor_mode == "pair_with_each_current" and len(current_slots) > 0:
                for cur_idx, (cur_symptom, cur_value) in enumerate(current_slots.items()):
                    split_current = OrderedDict()
                    split_current[cur_symptom] = cur_value
                    split_current[rare_symptom] = rare_value
                    split_current_variants.append((f"pair{cur_idx}", split_current))
            else:
                split_current = OrderedDict(current_slots)
                split_current[rare_symptom] = rare_value
                split_current_variants.append(("full", split_current))

            split_asked = OrderedDict(asked_implicit_slots)
            split_asked[rare_symptom] = rare_value

            rare_split_shuffle_num = max(1, int(args.rare_split_shuffle_num))
            rare_split_repeat_num = max(1, int(args.rare_split_repeat_num))

            for variant_name, split_current in split_current_variants:
                for shuffle_index in range(rare_split_shuffle_num):
                    split_targets_items = list(other_items)
                    if shuffle_index > 0 and len(split_targets_items) > 1:
                        rng.shuffle(split_targets_items)
                    split_targets = make_ordered_slots_from_items(split_targets_items)

                    # Keep position repeat strength, multiply by rare_split_repeat_num.
                    total_repeat = max(1, int(repeat_count)) * rare_split_repeat_num
                    for repeat_index in range(total_repeat):
                        sample = build_multi_positive_sample(
                            case=case,
                            current_slots=split_current,
                            asked_implicit_slots=split_asked,
                            remaining_implicit_slots=split_targets,
                            vocab=vocab,
                            sample_id=(
                                f"{sample_prefix}_raresplit_{variant_name}_r{rare_pos}_{rare_id_part}_"
                                f"shuf{shuffle_index}_rep{repeat_index}"
                            ),
                            step=step,
                            order_index=order_index,
                            target_shuffle_index=shuffle_index,
                            position_repeat_count=total_repeat,
                            position_repeat_index=repeat_index,
                            augment_type="rare_split",
                            rare_implicit_token=rare_token,
                            rare_aug_index=rare_aug_index,
                        )
                        augmented.append(sample)
                        rare_aug_index += 1

    return augmented


def expand_case_by_order(case, vocab, implicit_order, order_index=0, rng=None, rare_implicit_tokens=None):
    if rng is None:
        rng = random.Random(args.seed)
    if rare_implicit_tokens is None:
        rare_implicit_tokens = set()

    goal = case.get("goal", {})
    explicit = to_ordered_dict(goal.get("explicit_inform_slots", {}), include_false=bool(args.include_false_explicit))

    consult_id = case.get("consult_id", "")
    expanded = []

    m = len(implicit_order)
    for step in range(m):
        current_slots = OrderedDict(explicit)
        asked_implicit_slots = OrderedDict()

        for j in range(step):
            s, v = implicit_order[j]
            current_slots[s] = v
            asked_implicit_slots[s] = v

        remaining_items = []
        for j in range(step, m):
            s, v = implicit_order[j]
            remaining_items.append((s, v))

        repeat_count = compute_position_repeat_count(step=step, num_implicit=m)
        target_shuffle_num = max(1, int(args.target_shuffle_num))
        effective_shuffle_num = 1 if len(remaining_items) <= 1 else target_shuffle_num

        for shuffle_index in range(effective_shuffle_num):
            remaining_implicit_slots = shuffled_ordered_dict(
                items=remaining_items,
                rng=rng,
                shuffle_index=shuffle_index
            )

            for repeat_index in range(repeat_count):
                sample_id = (
                    f"{consult_id}_ord{order_index}_step{step}_"
                    f"shuf{shuffle_index}_rep{repeat_index}"
                )

                sample = build_multi_positive_sample(
                    case=case,
                    current_slots=current_slots,
                    asked_implicit_slots=asked_implicit_slots,
                    remaining_implicit_slots=remaining_implicit_slots,
                    vocab=vocab,
                    sample_id=sample_id,
                    step=step,
                    order_index=order_index,
                    target_shuffle_index=shuffle_index,
                    position_repeat_count=repeat_count,
                    position_repeat_index=repeat_index,
                    augment_type="base",
                )

                expanded.append(sample)

        # Rare implicit targeted augmentation: per step, use unshuffled remaining_items to avoid over-multiplying with base shuffle.
        sample_prefix = f"{consult_id}_ord{order_index}_step{step}"
        expanded.extend(build_rare_implicit_aug_samples(
            case=case,
            current_slots=current_slots,
            asked_implicit_slots=asked_implicit_slots,
            remaining_items=remaining_items,
            vocab=vocab,
            sample_prefix=sample_prefix,
            step=step,
            order_index=order_index,
            rng=rng,
            repeat_count=repeat_count,
            rare_implicit_tokens=rare_implicit_tokens,
        ))

    return expanded


def expand_case_single_order(case, vocab, rare_implicit_tokens=None):
    goal = case.get("goal", {})
    implicit = to_ordered_dict(goal.get("implicit_inform_slots", {}), include_false=bool(args.include_false_implicit))
    implicit_order = list(implicit.items())

    if len(implicit_order) == 0:
        return []

    consult_id = case.get("consult_id", "")
    rng = random.Random(f"{args.seed}_{consult_id}_single")

    return expand_case_by_order(
        case=case,
        vocab=vocab,
        implicit_order=implicit_order,
        order_index=0,
        rng=rng,
        rare_implicit_tokens=rare_implicit_tokens,
    )


def expand_case_repeated_permutation(case, vocab, repeat_num=4, seed=42, deduplicate_states=True, rare_implicit_tokens=None):
    goal = case.get("goal", {})
    implicit = to_ordered_dict(goal.get("implicit_inform_slots", {}), include_false=bool(args.include_false_implicit))
    implicit_items = list(implicit.items())

    if len(implicit_items) == 0:
        return []

    consult_id = case.get("consult_id", "")
    local_seed = f"{seed}_{consult_id}"
    rng = random.Random(local_seed)

    expanded = []
    seen_states = set()

    for order_index in range(repeat_num):
        cur_order = list(implicit_items)
        if order_index > 0:
            rng.shuffle(cur_order)

        order_samples = expand_case_by_order(
            case=case,
            vocab=vocab,
            implicit_order=cur_order,
            order_index=order_index,
            rng=rng,
            rare_implicit_tokens=rare_implicit_tokens,
        )

        for sample in order_samples:
            key = (
                state_key_from_tokens(sample["current_tokens"])
                + f"||targets={'-'.join(map(str, sample['positive_target_ids']))}"
                + f"||step={sample['step']}"
                + f"||order={sample['order_index']}"
                + f"||shuf={sample['target_shuffle_index']}"
                + f"||rep={sample['position_repeat_index']}"
                + f"||aug={sample.get('augment_type', 'base')}"
                + f"||rare={sample.get('rare_implicit_token', '')}"
                + f"||id={sample.get('sample_id', '')}"
            )

            if deduplicate_states and key in seen_states:
                continue

            seen_states.add(key)
            expanded.append(sample)

    return expanded


def expand_case_all_subsets(case, vocab, max_implicit_for_all_subsets=12, rare_implicit_tokens=None):
    if rare_implicit_tokens is None:
        rare_implicit_tokens = set()

    goal = case.get("goal", {})
    explicit = to_ordered_dict(goal.get("explicit_inform_slots", {}), include_false=bool(args.include_false_explicit))
    implicit = to_ordered_dict(goal.get("implicit_inform_slots", {}), include_false=bool(args.include_false_implicit))
    implicit_items = list(implicit.items())

    m = len(implicit_items)
    if m == 0:
        return []

    if m > max_implicit_for_all_subsets:
        raise ValueError(
            f"consult_id={case.get('consult_id', '')} has {m} implicit symptoms, "
            f"exceeding max_implicit_for_all_subsets={max_implicit_for_all_subsets}, "
            f"all_subsets may cause combinatorial explosion."
        )

    consult_id = case.get("consult_id", "")
    rng = random.Random(f"{args.seed}_{consult_id}_subsets")
    expanded = []

    for k in range(m):
        for asked_indices in combinations(range(m), k):
            asked_set = set(asked_indices)

            current_slots = OrderedDict(explicit)
            asked_implicit_slots = OrderedDict()

            for idx in asked_indices:
                s, v = implicit_items[idx]
                current_slots[s] = v
                asked_implicit_slots[s] = v

            remaining_items = []
            for idx, (s, v) in enumerate(implicit_items):
                if idx not in asked_set:
                    remaining_items.append((s, v))

            repeat_count = compute_position_repeat_count(step=k, num_implicit=m)
            target_shuffle_num = max(1, int(args.target_shuffle_num))
            effective_shuffle_num = 1 if len(remaining_items) <= 1 else target_shuffle_num

            for shuffle_index in range(effective_shuffle_num):
                remaining_implicit_slots = shuffled_ordered_dict(
                    items=remaining_items,
                    rng=rng,
                    shuffle_index=shuffle_index
                )

                for repeat_index in range(repeat_count):
                    sample_id = (
                        f"{consult_id}_subset_k{k}_"
                        f"a{'-'.join(map(str, asked_indices)) if asked_indices else 'none'}_"
                        f"shuf{shuffle_index}_rep{repeat_index}"
                    )

                    sample = build_multi_positive_sample(
                        case=case,
                        current_slots=current_slots,
                        asked_implicit_slots=asked_implicit_slots,
                        remaining_implicit_slots=remaining_implicit_slots,
                        vocab=vocab,
                        sample_id=sample_id,
                        step=k,
                        order_index=-1,
                        target_shuffle_index=shuffle_index,
                        position_repeat_count=repeat_count,
                        position_repeat_index=repeat_index,
                        augment_type="base",
                    )

                    expanded.append(sample)

            sample_prefix = (
                f"{consult_id}_subset_k{k}_"
                f"a{'-'.join(map(str, asked_indices)) if asked_indices else 'none'}"
            )
            expanded.extend(build_rare_implicit_aug_samples(
                case=case,
                current_slots=current_slots,
                asked_implicit_slots=asked_implicit_slots,
                remaining_items=remaining_items,
                vocab=vocab,
                sample_prefix=sample_prefix,
                step=k,
                order_index=-1,
                rng=rng,
                repeat_count=repeat_count,
                rare_implicit_tokens=rare_implicit_tokens,
            ))

    return expanded


def expand_cases(cases, vocab, rare_implicit_tokens=None):
    expanded = []
    skipped_no_implicit = 0

    if rare_implicit_tokens is None:
        rare_implicit_tokens = set()

    for case in tqdm(cases, desc="Expanding cases"):
        goal = case.get("goal", {})
        implicit = to_ordered_dict(goal.get("implicit_inform_slots", {}), include_false=bool(args.include_false_implicit))
        m = len(implicit)

        if m == 0:
            skipped_no_implicit += 1
            continue

        if args.mode == "single_order":
            samples = expand_case_single_order(case, vocab, rare_implicit_tokens=rare_implicit_tokens)

        elif args.mode == "repeated_permutation":
            samples = expand_case_repeated_permutation(
                case=case,
                vocab=vocab,
                repeat_num=args.repeat_num,
                seed=args.seed,
                deduplicate_states=bool(args.deduplicate_states),
                rare_implicit_tokens=rare_implicit_tokens,
            )

        elif args.mode == "all_subsets":
            samples = expand_case_all_subsets(
                case=case,
                vocab=vocab,
                max_implicit_for_all_subsets=args.max_implicit_for_all_subsets,
                rare_implicit_tokens=rare_implicit_tokens,
            )

        else:
            raise ValueError("mode must be single_order, repeated_permutation, or all_subsets")

        expanded.extend(samples)

    return expanded, skipped_no_implicit


def calc_case_details(cases, expanded_samples):
    consult_to_count = {}
    consult_to_aug_count = {}
    for s in expanded_samples:
        cid = s.get("consult_id", "")
        consult_to_count[cid] = consult_to_count.get(cid, 0) + 1
        if s.get("augment_type", "base") != "base":
            consult_to_aug_count[cid] = consult_to_aug_count.get(cid, 0) + 1

    details = []
    for case in cases:
        goal = case.get("goal", {})
        consult_id = case.get("consult_id", "")
        num_implicit = len(to_ordered_dict(goal.get("implicit_inform_slots", {}), include_false=bool(args.include_false_implicit)))
        details.append({
            "consult_id": consult_id,
            "num_implicit": num_implicit,
            "num_expanded_samples": consult_to_count.get(consult_id, 0),
            "num_rare_aug_samples": consult_to_aug_count.get(consult_id, 0)
        })

    return details


# =========================================================
# Extra DiffDiag matrix export
# =========================================================

def _special_ids(vocab):
    return {vocab.get("[PAD]"), vocab.get("[UNK]"), vocab.get("[MASK]")}


def ids_to_multihot(ids, vocab_size, skip_ids):
    vec = np.zeros(vocab_size, dtype=np.float32)
    for i in ids:
        if i in skip_ids:
            continue
        if 0 <= int(i) < vocab_size:
            vec[int(i)] = 1.0
    return vec


def samples_to_arrays(samples, vocab):
    vocab_size = len(vocab)
    skip_ids = _special_ids(vocab)

    obs_list, full_list, target_list, weight_list = [], [], [], []
    kept_records = []
    dropped_empty_target = 0

    for sample in samples:
        obs = ids_to_multihot(sample["current_token_ids"], vocab_size, skip_ids)
        target = ids_to_multihot(sample["positive_target_ids"], vocab_size, skip_ids)
        if target.sum() <= 0:
            dropped_empty_target += 1
            continue

        # x_full means current known + remaining hidden under this state.
        # This matches the original expanded sample semantics.
        full = np.maximum(obs, target).astype(np.float32)

        weight = np.ones(vocab_size, dtype=np.float32) * args.neg_weight
        weight[full > 0] = args.other_pos_weight
        weight[obs > 0] = args.obs_weight
        weight[target > 0] = args.hidden_weight

        obs_list.append(obs)
        full_list.append(full)
        target_list.append(target)
        weight_list.append(weight)
        kept_records.append(sample)

    if len(obs_list) == 0:
        raise ValueError("No valid samples generated. Check include_false / vocab_from / raw data.")

    return (
        np.stack(obs_list).astype(np.float32),
        np.stack(full_list).astype(np.float32),
        np.stack(target_list).astype(np.float32),
        np.stack(weight_list).astype(np.float32),
        kept_records,
        dropped_empty_target,
    )


def save_arrays(out_dir, split, arrays):
    obs, full, target, weight, kept_records, _ = arrays
    np.save(os.path.join(out_dir, f"{split}_obs.npy"), obs)
    np.save(os.path.join(out_dir, f"{split}_full.npy"), full)
    np.save(os.path.join(out_dir, f"{split}_target.npy"), target)
    np.save(os.path.join(out_dir, f"{split}_weight.npy"), weight)
    save_jsonl(kept_records, os.path.join(out_dir, f"{split}_samples_for_diffdiag.jsonl"))


def build_item_emb(train_full, emb_dim):
    # Deterministic co-occurrence SVD embedding for DiffDiag clustering.
    X = train_full.astype(np.float32)
    co = X.T @ X
    co = co + np.eye(co.shape[0], dtype=np.float32) * 1e-6
    u, s, _ = np.linalg.svd(co, full_matrices=False)
    dim = min(emb_dim, u.shape[1])
    emb = u[:, :dim] * np.sqrt(s[:dim])[None, :]
    if dim < emb_dim:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], emb_dim - dim), dtype=np.float32)], axis=1)
    return emb.astype(np.float32)


# =========================================================
# GMD format data loading and conversion
# =========================================================

def convert_gmd_case(gmd_case):
    """Convert GMD-format case to internal format.

    GMD format:
      - self_report: dominant -> explicit_inform_slots
      - all_symptoms - self_report: recessive -> implicit_inform_slots
    """
    self_report = gmd_case.get("self_report", {})
    all_symptoms = gmd_case.get("all_symptoms", {})

    implicit_slots = {}
    for symptom, value in all_symptoms.items():
        if symptom not in self_report:
            implicit_slots[symptom] = value

    return {
        "consult_id": str(gmd_case.get("id", "")),
        "disease_tag": gmd_case.get("disease_label", ""),
        "goal": {
            "explicit_inform_slots": dict(self_report),
            "implicit_inform_slots": implicit_slots,
            "origin_inform_slots": list(self_report.keys()),
        },
        "self_repo": "",
        "dialog": [],
    }


def load_gmd_cases(gmd_data_dir):
    """Load pre-split train / dev / test from GMD data directory and convert.

    Returns:
        train_cases, valid_cases, test_cases
    """
    train_cases = []
    valid_cases = []
    test_cases = []

    train_path = os.path.join(gmd_data_dir, "train.json")
    dev_path = os.path.join(gmd_data_dir, "dev.json")
    test_path = os.path.join(gmd_data_dir, "test.json")

    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        train_cases = [convert_gmd_case(c) for c in raw]
        print(f"Loaded {len(train_cases)} train cases from {train_path}")

    if os.path.exists(dev_path):
        with open(dev_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        valid_cases = [convert_gmd_case(c) for c in raw]
        print(f"Loaded {len(valid_cases)} valid cases from {dev_path}")

    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        test_cases = [convert_gmd_case(c) for c in raw]
        print(f"Loaded {len(test_cases)} test cases from {test_path}")

    return train_cases, valid_cases, test_cases


# =========================================================
# Main
# =========================================================

def main():
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- GMD data format: prefer pre-split train/dev/test ----
    gmd_data_dir = args.gmd_data_dir.strip() if args.gmd_data_dir else ""
    if gmd_data_dir and os.path.isdir(gmd_data_dir):
        train_cases, valid_cases, test_cases = load_gmd_cases(gmd_data_dir)
        split_source = "gmd_data"
        has_valid = len(valid_cases) > 0
    else:
        train_cases, test_cases, split_source = load_cases(args.input)
        valid_cases = []
        has_valid = False

    print(f"split_source: {split_source}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Valid cases: {len(valid_cases)}")
    print(f"Test cases: {len(test_cases)}")
    if not has_valid:
        print("Validation set: split at training time by DiffDiag/main.py from train samples.")

    vocab_cases = train_cases if args.vocab_from == "train" else train_cases + test_cases
    vocab = build_token_vocab(vocab_cases)
    id2token = {str(v): k for k, v in vocab.items()}

    print(f"Token vocab size: {len(vocab)}")
    print(f"[PAD] id: {vocab['[PAD]']}")
    print(f"[UNK] id: {vocab['[UNK]']}")
    print(f"[MASK] id: {vocab['[MASK]']}")

    # Count rare implicit symptoms from train set only, avoid leaking test distribution into augmentation.
    if bool(args.rare_implicit_augment):
        rare_implicit_info = compute_rare_implicit_symptoms(
            train_cases,
            vocab=vocab,
            topk=args.rare_implicit_topk
        )
    else:
        rare_implicit_info = []

    train_rare_implicit_tokens = {x["token"] for x in rare_implicit_info}
    # Valid set uses same rare symptom set as train (consistent, no test leakage)
    valid_rare_implicit_tokens = train_rare_implicit_tokens if has_valid else set()
    test_rare_implicit_tokens = train_rare_implicit_tokens if bool(args.apply_rare_aug_to_test) else set()

    print(f"Rare implicit augment: {bool(args.rare_implicit_augment)}")
    print(f"Rare implicit topk: {args.rare_implicit_topk}")
    if rare_implicit_info:
        print("Rare implicit top10/token:", ", ".join([f"{x['token']}({x['count']})" for x in rare_implicit_info]))
    else:
        print("Rare implicit top10/token: []")

    train_expanded, train_skipped = expand_cases(
        train_cases,
        vocab,
        rare_implicit_tokens=train_rare_implicit_tokens
    )
    test_expanded, test_skipped = expand_cases(
        test_cases,
        vocab,
        rare_implicit_tokens=test_rare_implicit_tokens
    )

    train_arrays = samples_to_arrays(train_expanded, vocab)
    test_arrays = samples_to_arrays(test_expanded, vocab)

    save_json(vocab, os.path.join(args.out_dir, "token_vocab.json"))
    save_json(id2token, os.path.join(args.out_dir, "id2token.json"))

    save_arrays(args.out_dir, "train", train_arrays)
    save_arrays(args.out_dir, "test", test_arrays)

    item_emb = build_item_emb(train_arrays[1], args.item_emb_dim)
    np.save(os.path.join(args.out_dir, "item_emb.npy"), item_emb)

    print(f"Saved to {args.out_dir}")
    print(f"  train samples: {train_arrays[0].shape[0]}, test samples: {test_arrays[0].shape[0]}")
    print(f"  vocab size: {len(vocab)}, item_emb: {list(item_emb.shape)}")
    print(f"  train npy: obs={train_arrays[0].shape}, full={train_arrays[1].shape}, target={train_arrays[2].shape}")
    print(f"  test  npy: obs={test_arrays[0].shape}, full={test_arrays[1].shape}, target={test_arrays[2].shape}")


if __name__ == "__main__":
    main()
