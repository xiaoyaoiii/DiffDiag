#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_OK = True
except Exception:
    PCA = None
    SKLEARN_OK = False


# ============================================================
# Utility functions
# ============================================================

def load_json(path: str) -> Any:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad jsonl: {path}:{line_no}: {e}") from e
    return rows


def read_turn_file(path: str) -> List[Dict[str, Any]]:
    """Supports .json (list) and .jsonl."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".jsonl"):
        return read_jsonl(path)
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"turn data must be a list, got {type(data)} from {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_from_str(s: str):
    s = str(s).lower()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["fp32", "float32"]:
        return torch.float32
    return "auto"


def stable_unique(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        if x is None:
            continue
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def torch_load_compat(path: str, map_location="cpu"):
    """Compatible with PyTorch 2.6+ weights_only default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ============================================================
# Load labels / turn samples
# ============================================================

def load_disease_names(cfg: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    """
    Label priority:
    1. paths.label_file / paths.disease_corpus_file
       - txt: one disease per line
       - json list: ["disease1", ...]
       - json dict: {"disease_names": [...]} or {"label2id": {...}}
    2. Infer from candidate_diseases + probe_label/disease_label in turn data.
    """
    paths = cfg.get("paths", {})
    label_file = paths.get("label_file") or paths.get("disease_corpus_file")
    if label_file:
        label_file = os.path.expanduser(label_file)
        if not os.path.exists(label_file):
            raise FileNotFoundError(label_file)
        if label_file.lower().endswith(".txt"):
            with open(label_file, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            return stable_unique(labels)
        data = load_json(label_file)
        if isinstance(data, list):
            return stable_unique([str(x) for x in data])
        if isinstance(data, dict):
            if "disease_names" in data:
                return stable_unique([str(x) for x in data["disease_names"]])
            if "labels" in data:
                return stable_unique([str(x) for x in data["labels"]])
            if "label2id" in data:
                items = sorted(data["label2id"].items(), key=lambda kv: int(kv[1]))
                return [str(k) for k, _ in items]
        raise ValueError(f"Unsupported label_file format: {label_file}")

    labels = []
    for r in rows:
        labels.extend(r.get("candidate_diseases", []) or [])
        labels.append(r.get("probe_label") or r.get("disease_label"))
    return stable_unique(labels)


def make_fallback_probe_group_id(row: Dict[str, Any]) -> str:
    dataset_name = row.get("dataset_name", "DATA")
    sample_id = row.get("id", row.get("sample_id", "unknown"))
    return f"{dataset_name}-{sample_id}"


def make_fallback_probe_uid(row: Dict[str, Any]) -> str:
    group_id = row.get("probe_group_id") or make_fallback_probe_group_id(row)
    turn_id = row.get("turn_id", "na")
    aug_depth = row.get("aug_depth", "na")
    aug_repeat_id = row.get("aug_repeat_id", "na")
    state_key = row.get("probe_state_key") or ""
    return f"{group_id}|turn={turn_id}|depth={aug_depth}|repeat={aug_repeat_id}|state={state_key}"


def normalize_turn_row(row: Dict[str, Any], label2id: Dict[str, int]) -> Dict[str, Any]:
    label_name = row.get("probe_label") or row.get("disease_label")
    if label_name is None:
        raise ValueError(f"Missing probe_label/disease_label in row: {row}")
    label_name = str(label_name).strip()
    if label_name not in label2id:
        raise ValueError(
            f"Label {label_name!r} not in label2id. "
            f"Please check label_file or disease_corpus_file."
        )

    group_id = row.get("probe_group_id") or make_fallback_probe_group_id(row)
    uid = row.get("probe_uid") or make_fallback_probe_uid(row)

    new_row = dict(row)
    new_row["probe_group_id"] = str(group_id)
    new_row["probe_uid"] = str(uid)
    new_row["probe_label"] = label_name
    new_row["label_id"] = int(label2id[label_name])
    return new_row


# ============================================================
# turn -> prompt_messages
# ============================================================

def join_or_none(items: List[str]) -> str:
    items = [str(x) for x in (items or []) if str(x).strip()]
    return ", ".join(items) if items else "none"


def safe_format_template(template: str, values: Dict[str, Any]) -> str:
    """Format template string with placeholder support."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + str(key) + "}", str(value))
    return out


def symptoms_dict_to_pos_neg(symptoms: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    pos, neg = [], []
    for symptom, status in (symptoms or {}).items():
        if bool(status):
            pos.append(str(symptom))
        else:
            neg.append(str(symptom))
    return pos, neg


def get_prompt_disease_set(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """
    disease_set_source:
      - all: use full disease set from disease_corpurs.txt (for raw test.json)
      - row: use per-turn candidate_diseases (for DDO turn training format)
    """
    pcfg = cfg.get("prompt", {})
    source = pcfg.get("disease_set_source", "row")
    if source == "all":
        labels = cfg.get("_disease_names", []) or []
        if labels:
            return list(labels)
    return list(row.get("candidate_diseases", []) or cfg.get("_disease_names", []) or [])


def build_turn_prompt_messages(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert DDO turn / raw test case to chat prompt.
    Uses symptoms and disease set only; disease_label/probe_label NOT written to prompt.
    """
    pcfg = cfg.get("prompt", {})
    symptom_source = pcfg.get("symptom_source", "current")

    if row.get("source_type") == "raw_case":
        pos = row.get("current_turn_True", []) or []
        neg = row.get("current_turn_False", []) or []
    elif symptom_source == "last":
        pos = row.get("last_turn_True", []) or []
        neg = row.get("last_turn_False", []) or []
    elif symptom_source == "both":
        pos = row.get("current_turn_True", []) or []
        neg = row.get("current_turn_False", []) or []
    else:
        pos = row.get("current_turn_True", []) or []
        neg = row.get("current_turn_False", []) or []

    disease_set = get_prompt_disease_set(row, cfg)
    disease_set_text = join_or_none(disease_set)

    system_prompt = pcfg.get("system_prompt", "")

    default_template = (
        "你是一位经验丰富的医学专家，现在为你提供如下信息：\n\n"
        "## 患者的症状表现情况\n"
        "- 患者存在的症状：{positive_symptoms}。\n"
        "- 患者不存在的症状：{negative_symptoms}。\n\n"
        "## 疾病'{candidate_disease}'的症状知识\n"
        "请你根据患者的症状表现情况来判断是{疾病集合}中的哪一个。请直接输出判断结果，不要输出其它内容。"
    )

    if symptom_source == "both" and row.get("source_type") != "raw_case":
        last_pos = join_or_none(row.get("last_turn_True", []) or [])
        last_neg = join_or_none(row.get("last_turn_False", []) or [])
        template = pcfg.get("user_template_both") or pcfg.get("user_template") or default_template
        values = {
            "last_positive_symptoms": last_pos,
            "last_negative_symptoms": last_neg,
            "positive_symptoms": join_or_none(pos),
            "negative_symptoms": join_or_none(neg),
            "candidate_disease": disease_set_text,
            "candidate_diseases": disease_set_text,
            "disease_set": disease_set_text,
            "disease_set": disease_set_text,
            "disease_set": disease_set_text,
            "疾病集合": disease_set_text,
        }
    else:
        template = pcfg.get("user_template") or default_template
        values = {
            "positive_symptoms": join_or_none(pos),
            "negative_symptoms": join_or_none(neg),
            "candidate_disease": disease_set_text,
            "candidate_diseases": disease_set_text,
            "disease_set": disease_set_text,
            "disease_set": disease_set_text,
            "disease_set": disease_set_text,
            "疾病集合": disease_set_text,
        }

    user_prompt = safe_format_template(template, values)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


# ============================================================
# Group split: prevent augmented turn leakage
# ============================================================

def split_rows_by_group(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split by probe_group_id. Stratified by probe_label to preserve disease distribution.
    Groups with too few samples are kept in train.
    """
    rng = random.Random(seed)
    group_to_rows = defaultdict(list)
    group_to_label = {}

    for r in rows:
        gid = r["probe_group_id"]
        group_to_rows[gid].append(r)
        label = r["probe_label"]
        if gid in group_to_label and group_to_label[gid] != label:
            raise ValueError(
                f"A probe_group_id has multiple labels: {gid}, "
                f"{group_to_label[gid]} vs {label}"
            )
        group_to_label[gid] = label

    label_to_groups = defaultdict(list)
    for gid, label in group_to_label.items():
        label_to_groups[label].append(gid)

    train_groups, val_groups, test_groups = set(), set(), set()

    for label, gids in label_to_groups.items():
        gids = list(gids)
        rng.shuffle(gids)
        n = len(gids)

        if n <= 1:
            train_groups.update(gids)
            continue

        n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0

        if n >= 3:
            if test_ratio > 0 and n_test == 0:
                n_test = 1
            if val_ratio > 0 and n_val == 0:
                n_val = 1
            while n_test + n_val >= n:
                if n_val >= n_test and n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1
                else:
                    break
        elif n == 2:
            if test_ratio > 0:
                n_test, n_val = 1, 0
            elif val_ratio > 0:
                n_test, n_val = 0, 1
            else:
                n_test, n_val = 0, 0

        test_part = gids[:n_test]
        val_part = gids[n_test:n_test + n_val]
        train_part = gids[n_test + n_val:]

        test_groups.update(test_part)
        val_groups.update(val_part)
        train_groups.update(train_part)

    split_rows = {"train": [], "val": [], "test": []}
    for gid, rs in group_to_rows.items():
        if gid in test_groups:
            split_rows["test"].extend(rs)
        elif gid in val_groups:
            split_rows["val"].extend(rs)
        else:
            split_rows["train"].extend(rs)

    for split in split_rows:
        rng.shuffle(split_rows[split])

    return split_rows


def normalize_raw_case_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def convert_raw_case_to_probe_row(
    case: Dict[str, Any],
    label2id: Dict[str, int],
    disease_names: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert raw case format (id / disease_label / self_report / all_symptoms) to probe row.
    Default test uses test.symptoms_key = all_symptoms.
    """
    paths = cfg.get("paths", {})
    dataset_name = str(paths.get("dataset_name", case.get("dataset_name", "DATA")))
    symptoms_key = cfg.get("test", {}).get("symptoms_key", "all_symptoms")
    symptoms = case.get(symptoms_key)
    if symptoms is None:
        raise ValueError(f"Raw test case missing symptoms_key={symptoms_key!r}: {case}")
    if not isinstance(symptoms, dict):
        raise TypeError(f"Raw test case {symptoms_key} must be dict, got {type(symptoms)}")

    label_name = str(case.get("probe_label") or case.get("disease_label", "")).strip()
    if not label_name:
        raise ValueError(f"Raw test case missing disease_label/probe_label: {case}")
    if label_name not in label2id:
        raise ValueError(f"Raw test label {label_name!r} not in disease corpus / label2id.")

    pos, neg = [], []
    for symptom, value in symptoms.items():
        if normalize_raw_case_value(value):
            pos.append(str(symptom))
        else:
            neg.append(str(symptom))

    sample_id = case.get("id", case.get("sample_id", "unknown"))
    group_id = f"{dataset_name}-rawtest-{sample_id}"
    uid = f"{group_id}|symptoms_key={symptoms_key}|final_all_symptoms"

    row = dict(case)
    row.update({
        "source_type": "raw_case",
        "dataset_name": dataset_name,
        "probe_group_id": group_id,
        "probe_uid": uid,
        "probe_label": label_name,
        "label_id": int(label2id[label_name]),
        "turn_id": "raw_final",
        "aug_depth": "raw_final",
        "aug_repeat_id": 1,
        "added_symptoms": [],
        "branch_enabled": False,
        "repeat_enabled": False,
        "candidate_diseases": list(disease_names),
        "last_turn_True": [],
        "last_turn_False": [],
        "current_turn_True": pos,
        "current_turn_False": neg,
        "raw_symptoms_key": symptoms_key,
    })
    return row


def read_raw_case_file(path: str) -> List[Dict[str, Any]]:
    return read_turn_file(path)


def load_and_prepare_splits(cfg: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], List[str]]:
    paths = cfg["paths"]
    split_cfg = cfg.get("split", {})
    seed = int(cfg.get("seed", 42))

    test_data_file = paths.get("test_data_file")

    # 1) Explicit train/val/test turn files provided.
    if "turn_data_files" in paths and paths["turn_data_files"]:
        all_rows_for_labels = []
        raw_splits = {}
        for split, p in paths["turn_data_files"].items():
            raw_splits[split] = read_turn_file(p)
            all_rows_for_labels.extend(raw_splits[split])
        if test_data_file:
            all_rows_for_labels.extend(read_raw_case_file(test_data_file))
        disease_names = load_disease_names(cfg, all_rows_for_labels)
        cfg["_disease_names"] = disease_names
        label2id = {d: i for i, d in enumerate(disease_names)}
        splits = {
            split: [normalize_turn_row(r, label2id) for r in rs]
            for split, rs in raw_splits.items()
        }
        if test_data_file:
            raw_test_cases = read_raw_case_file(test_data_file)
            max_test_samples = int(cfg.get("test", {}).get("max_test_samples", -1))
            if max_test_samples > 0:
                raw_test_cases = raw_test_cases[:max_test_samples]
            splits["test"] = [
                convert_raw_case_to_probe_row(case, label2id, disease_names, cfg)
                for case in raw_test_cases
            ]
        return splits, label2id, disease_names

    # 2) Single turn_data_file, split internally by group.
    turn_data_file = paths.get("turn_data_file")
    if not turn_data_file:
        dataset_name = paths.get("dataset_name", "DXY")
        data_root = paths.get("data_root")
        if not data_root:
            raise ValueError("Please set paths.turn_data_file or paths.data_root + paths.dataset_name")
        turn_data_file = os.path.join(os.path.expanduser(data_root), "turn_data", f"{dataset_name}_train_turn.json")

    raw_rows = read_turn_file(turn_data_file)
    rows_for_labels = list(raw_rows)
    if test_data_file:
        rows_for_labels.extend(read_raw_case_file(test_data_file))

    disease_names = load_disease_names(cfg, rows_for_labels)
    cfg["_disease_names"] = disease_names
    label2id = {d: i for i, d in enumerate(disease_names)}
    rows = [normalize_turn_row(r, label2id) for r in raw_rows]

    if test_data_file:
        # When raw test.json exists, turn_data_file only splits train/val (no test extracted from it).
        train_val_splits = split_rows_by_group(
            rows,
            val_ratio=float(split_cfg.get("val_ratio", 0.1)),
            test_ratio=0.0,
            seed=seed,
        )
        raw_test_cases = read_raw_case_file(test_data_file)
        max_test_samples = int(cfg.get("test", {}).get("max_test_samples", -1))
        if max_test_samples > 0:
            raw_test_cases = raw_test_cases[:max_test_samples]
        test_rows = [
            convert_raw_case_to_probe_row(case, label2id, disease_names, cfg)
            for case in raw_test_cases
        ]
        return {
            "train": train_val_splits.get("train", []),
            "val": train_val_splits.get("val", []),
            "test": test_rows,
        }, label2id, disease_names

    splits = split_rows_by_group(
        rows,
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        test_ratio=float(split_cfg.get("test_ratio", 0.1)),
        seed=seed,
    )
    return splits, label2id, disease_names


def summarize_split(split_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out = {}
    for split, rows in split_rows.items():
        groups = sorted({r["probe_group_id"] for r in rows})
        labels = Counter(r["probe_label"] for r in rows)
        out[split] = {
            "num_rows": len(rows),
            "num_groups": len(groups),
            "label_counts": dict(sorted(labels.items(), key=lambda kv: kv[0])),
        }
    return out


# ============================================================
# Hidden state extractor
# ============================================================

class HiddenExtractor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_name_or_path = cfg["paths"]["model_name_or_path"]
        ecfg = cfg["extract"]
        self.device = torch.device(ecfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.max_length = int(ecfg.get("max_length", 2048))
        self.batch_size = int(ecfg.get("batch_size", 4))
        self.pooling = ecfg.get("pooling", "mean")
        self.layer_indices = ecfg.get("layer_indices", [-1])
        self.normalize = bool(ecfg.get("normalize", True))
        self.add_generation_prompt = bool(ecfg.get("add_generation_prompt", True))

        print(f"[Load tokenizer] {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.tokenizer.padding_side = ecfg.get("padding_side", "left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype_from_str(ecfg.get("torch_dtype", "bf16")),
        }
        if bool(ecfg.get("use_4bit", False)):
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = ecfg.get("device_map", "auto")
        else:
            if ecfg.get("device_map"):
                model_kwargs["device_map"] = ecfg.get("device_map")

        print(f"[Load model] {self.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        if not bool(ecfg.get("use_4bit", False)) and not ecfg.get("device_map"):
            self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model, "model"):
            self.backbone = self.model.model
        else:
            self.backbone = self.model

    def render(self, prompt_messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )

    def pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if self.pooling == "last_token":
            seq_len = attention_mask.shape[1]
            arange = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
            pos = arange.masked_fill(attention_mask == 0, -1)
            last_idx = pos.max(dim=1).values
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]

        if self.pooling == "first_token":
            # left-padding compatible: take first non-pad token, not hidden[:, 0].
            seq_len = attention_mask.shape[1]
            arange = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
            pos = arange.masked_fill(attention_mask == 0, seq_len + 1)
            first_idx = pos.min(dim=1).values
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), first_idx]

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def select_hidden(self, hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_list = []
        n = len(hidden_states)
        for idx in self.layer_indices:
            real_idx = idx if idx >= 0 else n + idx
            if real_idx < 0 or real_idx >= n:
                raise IndexError(f"layer index {idx} out of range, hidden_states len={n}")
            pooled = self.pool_hidden(hidden_states[real_idx], attention_mask)
            pooled_list.append(pooled)
        feat = torch.cat(pooled_list, dim=-1)
        if self.normalize:
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    @torch.no_grad()
    def encode_rows(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        all_feats = []
        for st in tqdm(range(0, len(rows), self.batch_size), desc="extract hidden"):
            batch = rows[st: st + self.batch_size]
            texts = [self.render(x["prompt_messages"]) for x in batch]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.backbone(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
            feat = self.select_hidden(out.hidden_states, enc["attention_mask"])
            all_feats.append(feat.detach().float().cpu().numpy())
            del enc, out, feat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(all_feats, axis=0).astype(np.float32)


# ============================================================
# Feature save / load
# ============================================================

def save_features(path: str, X: np.ndarray, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y = np.array([r["label_id"] for r in rows], dtype=np.int64)
    uid = np.array([r["probe_uid"] for r in rows], dtype=object)
    group_id = np.array([r["probe_group_id"] for r in rows], dtype=object)
    label_name = np.array([r["probe_label"] for r in rows], dtype=object)
    candidate_diseases = np.array([r.get("candidate_diseases", []) for r in rows], dtype=object)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=y,
        uid=uid,
        group_id=group_id,
        label_name=label_name,
        candidate_diseases=candidate_diseases,
    )


def load_npz(path: str):
    z = np.load(path, allow_pickle=True)
    return {
        "X": z["X"].astype(np.float32),
        "y": z["y"].astype(np.int64),
        "uid": z["uid"],
        "group_id": z["group_id"],
        "label_name": z["label_name"],
        "candidate_diseases": z["candidate_diseases"],
    }


def extract_all(cfg: Dict[str, Any]) -> None:
    feature_dir = cfg["paths"]["feature_dir"]
    Path(feature_dir).mkdir(parents=True, exist_ok=True)

    split_rows, label2id, disease_names = load_and_prepare_splits(cfg)
    id2label = {int(v): k for k, v in label2id.items()}

    split_summary = summarize_split(split_rows)
    save_json(os.path.join(feature_dir, "split_summary.json"), split_summary)
    save_json(os.path.join(feature_dir, "label_map.json"), {
        "label2id": label2id,
        "id2label": id2label,
        "disease_names": disease_names,
    })
    save_json(os.path.join(feature_dir, "feature_meta.json"), {
        "model": cfg["paths"]["model_name_or_path"],
        "extract": cfg["extract"],
        "prompt": cfg.get("prompt", {}),
        "num_labels": len(disease_names),
        "labels": disease_names,
        "split_summary": split_summary,
    })
    print("[Split summary]")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))

    extractor = HiddenExtractor(cfg)

    for split in cfg.get("extract", {}).get("splits", ["train", "val", "test"]):
        rows = split_rows.get(split, [])
        if not rows:
            print(f"[Skip empty split] {split}")
            continue

        out_path = os.path.join(feature_dir, f"features_{split}.npz")
        meta_path = os.path.join(feature_dir, f"meta_{split}.jsonl")
        if os.path.exists(out_path) and bool(cfg["extract"].get("resume", True)):
            print(f"[Skip exists] {out_path}")
            continue

        prepared_rows = []
        for r in rows:
            rr = dict(r)
            rr["prompt_messages"] = build_turn_prompt_messages(rr, cfg)
            prepared_rows.append(rr)

        print(f"[Extract] split={split}, n={len(prepared_rows)}")
        X = extractor.encode_rows(prepared_rows)
        save_features(out_path, X, prepared_rows)

        if os.path.exists(meta_path):
            os.remove(meta_path)
        for r in prepared_rows:
            append_jsonl(meta_path, {
                "probe_uid": r["probe_uid"],
                "probe_group_id": r["probe_group_id"],
                "probe_label": r["probe_label"],
                "label_id": r["label_id"],
                "dataset_name": r.get("dataset_name"),
                "id": r.get("id"),
                "turn_id": r.get("turn_id"),
                "aug_depth": r.get("aug_depth"),
                "aug_repeat_id": r.get("aug_repeat_id"),
                "added_symptoms": r.get("added_symptoms"),
                "branch_enabled": r.get("branch_enabled"),
                "repeat_enabled": r.get("repeat_enabled"),
                "candidate_diseases": r.get("candidate_diseases", []),
                "last_turn_True": r.get("last_turn_True", []),
                "last_turn_False": r.get("last_turn_False", []),
                "current_turn_True": r.get("current_turn_True", []),
                "current_turn_False": r.get("current_turn_False", []),
                "prompt_messages": r.get("prompt_messages"),
            })


# ============================================================
# PCA
# ============================================================

def fit_transform_pca(
    cfg: Dict[str, Any],
    X_train: np.ndarray,
    X_val: np.ndarray,
    probe_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    pcfg = cfg.get("pca", {})
    if not bool(pcfg.get("enabled", True)):
        meta = {
            "enabled": False,
            "input_dim": int(X_train.shape[1]),
            "output_dim": int(X_train.shape[1]),
        }
        save_json(os.path.join(probe_dir, "pca_meta.json"), meta)
        return X_train, X_val, meta

    if not SKLEARN_OK or PCA is None:
        raise RuntimeError("pca.enabled=True requires scikit-learn. Please install sklearn or set pca.enabled=false.")

    requested = int(pcfg.get("n_components", 256))
    max_allowed = min(X_train.shape[0], X_train.shape[1])
    n_components = min(requested, max_allowed)
    if n_components < requested:
        print(f"[PCA] n_components adjusted: {requested} -> {n_components} because max_allowed={max_allowed}")

    pca = PCA(
        n_components=n_components,
        whiten=bool(pcfg.get("whiten", False)),
        random_state=int(cfg.get("seed", 42)),
    )
    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_val_pca = pca.transform(X_val).astype(np.float32)

    pca_path = os.path.join(probe_dir, "pca.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    explained = getattr(pca, "explained_variance_ratio_", None)
    meta = {
        "enabled": True,
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(n_components),
        "requested_n_components": int(requested),
        "n_components": int(n_components),
        "whiten": bool(pcfg.get("whiten", False)),
        "explained_variance_ratio_sum": float(np.sum(explained)) if explained is not None else None,
        "pca_path": pca_path,
    }
    save_json(os.path.join(probe_dir, "pca_meta.json"), meta)
    print("[PCA meta]")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return X_train_pca, X_val_pca, meta


def transform_with_saved_pca(cfg: Dict[str, Any], X: np.ndarray, probe_dir: str) -> np.ndarray:
    pca_meta_path = os.path.join(probe_dir, "pca_meta.json")
    if not os.path.exists(pca_meta_path):
        return X
    pca_meta = load_json(pca_meta_path)
    if not pca_meta.get("enabled", False):
        return X
    pca_path = pca_meta.get("pca_path") or os.path.join(probe_dir, "pca.pkl")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    return pca.transform(X).astype(np.float32)


# ============================================================
# Probe classifiers: Linear / MLP
# ============================================================

class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DiseaseLinearProbe(nn.Module):
    """
    Multi-class logistic regression / linear probe.
    logits = W x + b, trained with CrossEntropyLoss.
    """
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.classifier(x)


class DiseaseMLP(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, x):
        return self.net(x)


def build_probe_model(
    probe_type: str,
    input_dim: int,
    num_labels: int,
    hidden_dim: int = 512,
    dropout: float = 0.1,
) -> nn.Module:
    probe_type = str(probe_type).lower()
    if probe_type in ["linear", "logreg", "logistic", "logistic_regression"]:
        return DiseaseLinearProbe(input_dim=input_dim, num_labels=num_labels)
    if probe_type == "mlp":
        return DiseaseMLP(input_dim=input_dim, num_labels=num_labels, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported probe.type={probe_type!r}. Use 'linear' or 'mlp'.")


def compute_class_weights(y: np.ndarray, num_labels: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_labels).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict_disease(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for st in range(0, len(X), batch_size):
            xb = torch.tensor(X[st: st + batch_size], dtype=torch.float32, device=device)
            logits = model(xb) / max(float(temperature), 1e-6)
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0).astype(np.float32)


def topk_accuracy(y_true: np.ndarray, prob: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, prob.shape[1])
    topk = np.argsort(-prob, axis=1)[:, :k]
    ok = [int(y_true[i] in set(topk[i].tolist())) for i in range(len(y_true))]
    return float(np.mean(ok))


def compute_disease_metrics(y_true: np.ndarray, prob: np.ndarray, topk_list: List[int]) -> Dict[str, Any]:
    pred = np.argmax(prob, axis=1).astype(np.int64)
    out = {}
    if SKLEARN_OK:
        out["accuracy"] = float(accuracy_score(y_true, pred))
        out["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        out["weighted_f1"] = float(f1_score(y_true, pred, average="weighted", zero_division=0))
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
        out["macro_precision"] = float(p)
        out["macro_recall"] = float(r)
        out["confusion_matrix"] = confusion_matrix(y_true, pred).tolist()
    else:
        out["accuracy"] = float((pred == y_true).mean())
        out["macro_f1"] = None
        out["weighted_f1"] = None

    for k in topk_list:
        out[f"top_{k}_accuracy"] = topk_accuracy(y_true, prob, int(k))
    return out


def load_label_map(feature_dir: str) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    label_map = load_json(os.path.join(feature_dir, "label_map.json"))
    label2id = {str(k): int(v) for k, v in label_map["label2id"].items()}
    id2label = {int(k): str(v) for k, v in label_map["id2label"].items()}
    disease_names = label_map.get("disease_names") or [id2label[i] for i in range(len(id2label))]
    return label2id, id2label, disease_names


def train_probe(cfg: Dict[str, Any]) -> None:
    """Train probe: use val for checkpoint selection, final eval on test."""
    feature_dir = cfg["paths"]["feature_dir"]
    probe_dir = cfg["paths"]["probe_dir"]
    Path(probe_dir).mkdir(parents=True, exist_ok=True)

    train_data = load_npz(os.path.join(feature_dir, "features_train.npz"))
    X_train, y_train = train_data["X"], train_data["y"]

    val_data = load_npz(os.path.join(feature_dir, "features_val.npz"))
    X_val, y_val = val_data["X"], val_data["y"]

    test_data = load_npz(os.path.join(feature_dir, "features_test.npz"))
    X_test, y_test = test_data["X"], test_data["y"]

    label2id, id2label, disease_names = load_label_map(feature_dir)
    num_labels = len(disease_names)

    pcfg = cfg.get("probe", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PCA fit on train only; val/test transform only.
    X_train, X_val, pca_meta = fit_transform_pca(cfg, X_train, X_val, probe_dir)
    X_test = transform_with_saved_pca(cfg, X_test, probe_dir)

    probe_type = str(pcfg.get("type", "mlp")).lower()
    hidden_dim = int(pcfg.get("hidden_dim", 512))
    dropout = float(pcfg.get("dropout", 0.1))
    model = build_probe_model(
        probe_type=probe_type,
        input_dim=X_train.shape[1],
        num_labels=num_labels,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    print(f"[Probe] type={probe_type}, input_dim={X_train.shape[1]}, num_labels={num_labels}")
    print(f"[Checkpoint selection] val.{pcfg.get('select_metric', 'macro_f1')}")

    class_weight_cfg = pcfg.get("class_weight", "auto")
    if class_weight_cfg == "auto":
        weight = compute_class_weights(y_train, num_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif class_weight_cfg in [None, "none", "None", False]:
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("probe.class_weight only supports 'auto' or 'none' in this script.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pcfg.get("lr", 3e-4)),
        weight_decay=float(pcfg.get("weight_decay", 0.01)),
    )

    train_loader = DataLoader(
        FeatureDataset(X_train, y_train),
        batch_size=int(pcfg.get("batch_size", 128)),
        shuffle=True,
        drop_last=False,
    )

    best_score = -1.0
    best_epoch = -1
    patience = int(pcfg.get("patience", 5))
    bad = 0
    history = []
    topk_list = [int(k) for k in pcfg.get("topk", [1, 3, 5])]
    select_metric = pcfg.get("select_metric", "macro_f1")
    temperature = float(pcfg.get("temperature", 1.0))

    for epoch in range(1, int(pcfg.get("epochs", 20)) + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"train {probe_type} probe epoch {epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(pcfg.get("max_grad_norm", 5.0)))
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / max(len(y_train), 1)

        val_prob = predict_disease(model, X_val, int(pcfg.get("batch_size", 128)), device, temperature=temperature)
        val_metrics = compute_disease_metrics(y_val, val_prob, topk_list)

        record = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

        score = val_metrics.get(select_metric)
        if score is None:
            score = val_metrics.get("accuracy", 0.0)

        if float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch
            bad = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "probe_type": probe_type,
                "input_dim": int(X_train.shape[1]),
                "num_labels": int(num_labels),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "label2id": label2id,
                "id2label": id2label,
                "disease_names": disease_names,
                "config": cfg,
                "pca_meta": pca_meta,
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                "select_metric": select_metric,
                "best_val_metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(probe_dir, "best_disease_probe.pt"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[Early stop] best_epoch={best_epoch}, best_val_{select_metric}={best_score}")
                break

    save_json(os.path.join(probe_dir, "train_history.json"), history)

    # ---- Final test set evaluation ----
    best_ckpt = torch_load_compat(os.path.join(probe_dir, "best_disease_probe.pt"), map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_prob = predict_disease(model, X_test, int(pcfg.get("batch_size", 128)), device, temperature=temperature)
    test_metrics = compute_disease_metrics(y_test, test_prob, topk_list)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_val_score"] = float(best_score)
    save_json(os.path.join(probe_dir, "test_metrics.json"), test_metrics)
    print(f"[Test metrics]")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] {os.path.join(probe_dir, 'best_disease_probe.pt')}")


def load_probe(probe_path: str, device: torch.device):
    ckpt = torch_load_compat(probe_path, map_location="cpu")
    probe_type = str(ckpt.get("probe_type", "mlp")).lower()

    model = build_probe_model(
        probe_type=probe_type,
        input_dim=int(ckpt["input_dim"]),
        num_labels=int(ckpt["num_labels"]),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def read_meta_jsonl(path: str) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def make_topk_items(prob: np.ndarray, id2label: Dict[int, str], top_k: int) -> List[Dict[str, Any]]:
    top_ids = np.argsort(-prob)[:top_k]
    return [
        {"disease": id2label[int(i)], "label_id": int(i), "probability": float(prob[int(i)])}
        for i in top_ids
    ]


def make_candidate_topk(
    prob: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    candidates: List[str],
) -> List[Dict[str, Any]]:
    items = []
    for d in candidates or []:
        if d not in label2id:
            continue
        i = label2id[d]
        items.append({"disease": d, "label_id": int(i), "probability": float(prob[i])})
    return sorted(items, key=lambda x: x["probability"], reverse=True)


def test_probe(cfg: Dict[str, Any]) -> None:
    feature_dir = cfg["paths"]["feature_dir"]
    probe_dir = cfg["paths"]["probe_dir"]
    output_file = cfg["paths"].get("test_output_file", os.path.join(probe_dir, "disease_predictions.jsonl"))

    pcfg = cfg.get("probe", {})
    tcfg = cfg.get("test", {})
    split = tcfg.get("split", "test")
    top_k = int(tcfg.get("top_k", 5))
    save_all_probabilities = bool(tcfg.get("save_all_probabilities", False))
    topk_metrics = [int(k) for k in pcfg.get("topk", [1, 3, 5])]
    temperature = float(tcfg.get("temperature", pcfg.get("temperature", 1.0)))

    data = load_npz(os.path.join(feature_dir, f"features_{split}.npz"))
    X, y = data["X"], data["y"]
    metas = read_meta_jsonl(os.path.join(feature_dir, f"meta_{split}.jsonl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_probe(os.path.join(probe_dir, "best_disease_probe.pt"), device)
    label2id = {str(k): int(v) for k, v in ckpt["label2id"].items()}
    id2label = {int(k): str(v) for k, v in ckpt["id2label"].items()}

    X = transform_with_saved_pca(cfg, X, probe_dir)
    probs = predict_disease(model, X, int(pcfg.get("batch_size", 128)), device, temperature=temperature)
    metrics = compute_disease_metrics(y, probs, topk_metrics)
    metrics["probe_type"] = str(ckpt.get("probe_type", pcfg.get("type", "mlp")))
    save_json(os.path.join(probe_dir, f"{split}_metrics.json"), metrics)
    print(f"[{split} metrics]")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    for i, prob in enumerate(probs):
        pred_id = int(np.argmax(prob))
        gold_id = int(y[i])
        meta = metas[i] if i < len(metas) else {}
        candidates = meta.get("candidate_diseases")
        item = {
            "id": i,
            "probe_uid": str(data["uid"][i]),
            "probe_group_id": str(data["group_id"][i]),
            "gold_label": id2label[gold_id],
            "gold_label_id": gold_id,
            "pred_label": id2label[pred_id],
            "pred_label_id": pred_id,
            "pred_probability": float(prob[pred_id]),
            "gold_probability": float(prob[gold_id]),
            "topk": make_topk_items(prob, id2label, top_k),
            "candidate_topk": make_candidate_topk(prob, id2label, label2id, candidates or []),
        }
        if save_all_probabilities:
            item["disease_probabilities"] = {
                id2label[j]: float(prob[j]) for j in range(len(prob))
            }
        item.update({
            "turn_meta": {
                "dataset_name": meta.get("dataset_name"),
                "case_id": meta.get("id"),
                "turn_id": meta.get("turn_id"),
                "aug_depth": meta.get("aug_depth"),
                "aug_repeat_id": meta.get("aug_repeat_id"),
                "added_symptoms": meta.get("added_symptoms"),
                "branch_enabled": meta.get("branch_enabled"),
                "repeat_enabled": meta.get("repeat_enabled"),
                "candidate_diseases": candidates,
                "last_turn_True": meta.get("last_turn_True"),
                "last_turn_False": meta.get("last_turn_False"),
                "current_turn_True": meta.get("current_turn_True"),
                "current_turn_False": meta.get("current_turn_False"),
            }
        })
        append_jsonl(output_file, item)

    print(f"[Saved predictions] {output_file}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None, choices=["extract", "train", "test", "all"])
    args = parser.parse_args()

    cfg = load_json(args.config)
    mode = args.mode or cfg.get("mode", "all")
    set_seed(int(cfg.get("seed", 42)))

    if mode in ["extract", "all"]:
        extract_all(cfg)
    if mode in ["train", "all"]:
        train_probe(cfg)
    if mode in ["test", "all"]:
        test_probe(cfg)


if __name__ == "__main__":
    main()
