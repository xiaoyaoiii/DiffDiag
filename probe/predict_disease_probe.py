#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_OK = True
except Exception:
    PCA = None
    SKLEARN_OK = False


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_turn_file(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".jsonl"):
        return read_jsonl(path)
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"turn data must be a list, got {type(data)} from {path}")
    return data


def dtype_from_str(s: str):
    s = str(s).lower()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["fp32", "float32"]:
        return torch.float32
    return "auto"


def torch_load_compat(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ═══════════════════════════════════════════════════════════════
# Probe models
# ═══════════════════════════════════════════════════════════════

class DiseaseLinearProbe(nn.Module):
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


def build_probe_model(probe_type: str, input_dim: int, num_labels: int,
                      hidden_dim: int = 512, dropout: float = 0.1) -> nn.Module:
    probe_type = str(probe_type).lower()
    if probe_type in ["linear", "logreg", "logistic", "logistic_regression"]:
        return DiseaseLinearProbe(input_dim=input_dim, num_labels=num_labels)
    if probe_type == "mlp":
        return DiseaseMLP(input_dim=input_dim, num_labels=num_labels,
                          hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported probe.type={probe_type!r}")


def load_probe(probe_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch_load_compat(probe_path, map_location="cpu")
    probe_type = str(ckpt.get("probe_type",
                    ckpt.get("config", {}).get("probe", {}).get("type", "mlp"))).lower()
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


# ═══════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════

def load_saved_pca_object(probe_dir: str):
    pca_meta_path = os.path.join(probe_dir, "pca_meta.json")
    if not os.path.exists(pca_meta_path):
        return None
    meta = load_json(pca_meta_path)
    if not meta.get("enabled", False):
        return None
    pca_path = meta.get("pca_path") or os.path.join(probe_dir, "pca.pkl")
    with open(pca_path, "rb") as f:
        return pickle.load(f)


def transform_with_saved_pca(cfg: Dict[str, Any], X: np.ndarray, probe_dir: str) -> np.ndarray:
    meta_path = os.path.join(probe_dir, "pca_meta.json")
    if not os.path.exists(meta_path):
        return X
    meta = load_json(meta_path)
    if not meta.get("enabled", False):
        return X
    pca = load_saved_pca_object(probe_dir)
    return pca.transform(X).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Hidden state extraction via LLM
# ═══════════════════════════════════════════════════════════════

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
            prompt_messages, tokenize=False,
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
            seq_len = attention_mask.shape[1]
            arange = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
            pos = arange.masked_fill(attention_mask == 0, seq_len + 1)
            first_idx = pos.min(dim=1).values
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), first_idx]
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def select_hidden(self, hidden_states: Tuple[torch.Tensor, ...],
                      attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_list = []
        n = len(hidden_states)
        for idx in self.layer_indices:
            real_idx = idx if idx >= 0 else n + idx
            if real_idx < 0 or real_idx >= n:
                raise IndexError(f"layer index {idx} out of range")
            pooled = self.pool_hidden(hidden_states[real_idx], attention_mask)
            pooled_list.append(pooled)
        feat = torch.cat(pooled_list, dim=-1)
        if self.normalize:
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    @torch.no_grad()
    def encode_rows(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        all_feats = []
        show_bar = len(rows) > self.batch_size
        for st in tqdm(range(0, len(rows), self.batch_size),
                       desc="extract hidden", disable=not show_bar):
            batch = rows[st: st + self.batch_size]
            texts = [self.render(x["prompt_messages"]) for x in batch]
            enc = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.backbone(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
            feat = self.select_hidden(out.hidden_states, enc["attention_mask"])
            all_feats.append(feat.detach().float().cpu().numpy())
            del enc, out, feat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(all_feats, axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Prompt building (shared between static & dialogue modes)
# ═══════════════════════════════════════════════════════════════

def join_or_none(items: List[str]) -> str:
    items = [str(x) for x in (items or []) if str(x).strip()]
    return ", ".join(items) if items else "none"


def safe_format_template(template: str, values: Dict[str, Any]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + str(key) + "}", str(value))
    return out


def get_prompt_disease_set(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    pcfg = cfg.get("prompt", {})
    source = pcfg.get("disease_set_source", "row")
    if source == "all":
        labels = cfg.get("_disease_names", []) or []
        if labels:
            return list(labels)
    return list(row.get("candidate_diseases", []) or cfg.get("_disease_names", []) or [])


def build_turn_prompt_messages(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    pcfg = cfg.get("prompt", {})
    symptom_source = pcfg.get("symptom_source", "current")

    if row.get("source_type") == "raw_case":
        pos = row.get("current_turn_True", []) or []
        neg = row.get("current_turn_False", []) or []
    elif symptom_source == "last":
        pos = row.get("last_turn_True", []) or []
        neg = row.get("last_turn_False", []) or []
    else:
        pos = row.get("current_turn_True", []) or []
        neg = row.get("current_turn_False", []) or []

    disease_set = get_prompt_disease_set(row, cfg)
    disease_set_text = join_or_none(disease_set)

    system_prompt = pcfg.get("system_prompt", "")
    template = pcfg.get("user_template") or (
        "你是一位经验丰富的医学专家，现在为你提供如下信息：\n\n"
        "## 患者的症状表现情况\n"
        "- 患者存在的症状：{positive_symptoms}。\n"
        "- 患者不存在的症状：{negative_symptoms}。\n\n"
        "## 疾病集合\n{疾病集合}\n\n"
        "请你根据患者的症状表现情况判断患者最可能属于疾病集合中的哪一个。请直接输出判断结果，不要输出其它内容。"
    )

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


# ═══════════════════════════════════════════════════════════════
# Prediction helpers
# ═══════════════════════════════════════════════════════════════

def predict_disease(model: nn.Module, X: np.ndarray, batch_size: int,
                    device: torch.device, temperature: float = 1.0) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for st in range(0, len(X), batch_size):
            xb = torch.tensor(X[st: st + batch_size], dtype=torch.float32, device=device)
            logits = model(xb) / max(float(temperature), 1e-6)
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0).astype(np.float32)


def make_topk_items(prob: np.ndarray, id2label: Dict[int, str], top_k: int) -> List[Dict[str, Any]]:
    top_ids = np.argsort(-prob)[:top_k]
    return [
        {"disease": id2label[int(i)], "label_id": int(i), "probability": float(prob[int(i)])}
        for i in top_ids
    ]


def topk_accuracy(y_true: np.ndarray, prob: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, prob.shape[1])
    topk = np.argsort(-prob, axis=1)[:, :k]
    ok = [int(y_true[i] in set(topk[i].tolist())) for i in range(len(y_true))]
    return float(np.mean(ok))


def compute_disease_metrics(y_true: np.ndarray, prob: np.ndarray,
                            topk_list: List[int]) -> Dict[str, Any]:
    pred = np.argmax(prob, axis=1).astype(np.int64)
    out = {}
    if SKLEARN_OK:
        out["accuracy"] = float(accuracy_score(y_true, pred))
        out["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        out["weighted_f1"] = float(f1_score(y_true, pred, average="weighted", zero_division=0))
        p, r, _, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
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


def load_label_map(feature_dir: str):
    label_map = load_json(os.path.join(feature_dir, "label_map.json"))
    label2id = {str(k): int(v) for k, v in label_map["label2id"].items()}
    id2label = {int(k): str(v) for k, v in label_map["id2label"].items()}
    disease_names = label_map.get("disease_names") or [id2label[i] for i in range(len(id2label))]
    return label2id, id2label, disease_names


# ═══════════════════════════════════════════════════════════════
# Static prediction (on pre-extracted features)
# ═══════════════════════════════════════════════════════════════

def run_static_predict(cfg: Dict[str, Any]) -> None:
    feature_dir = cfg["paths"]["feature_dir"]
    probe_dir = cfg["paths"]["probe_dir"]
    output_file = cfg["paths"].get("test_output_file",
                                    os.path.join(probe_dir, "disease_predictions.jsonl"))

    tcfg = cfg.get("test", {})
    split = tcfg.get("split", "test")
    top_k = int(tcfg.get("top_k", 5))
    save_all = bool(tcfg.get("save_all_probabilities", False))
    temperature = float(tcfg.get("temperature", cfg.get("probe", {}).get("temperature", 1.0)))
    pcfg = cfg.get("probe", {})

    # load features
    z = np.load(os.path.join(feature_dir, f"features_{split}.npz"), allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    uids = z["uid"]
    gids = z["group_id"]
    metas = read_jsonl(os.path.join(feature_dir, f"meta_{split}.jsonl"))

    # load probe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_probe(os.path.join(probe_dir, "best_disease_probe.pt"), device)
    id2label = {int(k): str(v) for k, v in ckpt["id2label"].items()}
    label2id = {str(k): int(v) for k, v in ckpt["label2id"].items()}

    # PCA transform
    X = transform_with_saved_pca(cfg, X, probe_dir)

    # predict
    probs = predict_disease(model, X, int(pcfg.get("batch_size", 128)), device, temperature)
    metrics = compute_disease_metrics(y, probs, [int(k) for k in cfg.get("probe", {}).get("topk", [1, 3, 5])])
    metrics["probe_type"] = str(ckpt.get("probe_type", pcfg.get("type", "mlp")))
    save_json(os.path.join(probe_dir, f"{split}_metrics.json"), metrics)
    print(f"[{split} metrics]")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # save predictions
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "a", encoding="utf-8") as fout:
        for i, prob in enumerate(probs):
            pred_id = int(np.argmax(prob))
            gold_id = int(y[i])
            meta = metas[i] if i < len(metas) else {}
            candidates = meta.get("candidate_diseases")
            item = {
                "id": i,
                "probe_uid": str(uids[i]),
                "probe_group_id": str(gids[i]),
                "gold_label": id2label[gold_id],
                "gold_label_id": gold_id,
                "pred_label": id2label[pred_id],
                "pred_label_id": pred_id,
                "pred_probability": float(prob[pred_id]),
                "gold_probability": float(prob[gold_id]),
                "topk": make_topk_items(prob, id2label, top_k),
            }
            if save_all:
                item["disease_probabilities"] = {id2label[j]: float(prob[j]) for j in range(len(prob))}
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Saved predictions] {output_file}")


# ═══════════════════════════════════════════════════════════════
# Dialogue prediction helpers
# ═══════════════════════════════════════════════════════════════

def _add_diffdiag_path(code_dir: str) -> None:
    code_dir = os.path.abspath(os.path.expanduser(code_dir))
    if not os.path.isdir(code_dir):
        raise FileNotFoundError(f"DiffDiag code dir not found: {code_dir}")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def normalize_token_vocab(vocab: Dict[Any, Any]) -> Dict[str, int]:
    if not isinstance(vocab, dict):
        raise ValueError("vocab must be a dict")
    if all(str(k).isdigit() for k in vocab.keys()):
        return {str(v): int(k) for k, v in vocab.items()}
    return {str(k): int(v) for k, v in vocab.items()}


def invert_token_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in vocab.items()}


def normalize_case_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "yes", "y", "1"}:
            return True
        if v in {"false", "f", "no", "n", "0"}:
            return False
    return bool(value)


def make_state_token(symptom: str, value: Any) -> str:
    return f"{symptom}__{'true' if normalize_case_value(value) else 'false'}"


def get_case_id(case: Dict[str, Any], index: int) -> str:
    return str(case.get("consult_id", case.get("id", case.get("sample_id", index))))


def get_case_label(case: Dict[str, Any]) -> str:
    return str(case.get("disease_label", case.get("disease_tag", case.get("label", "")))).strip()


def get_case_explicit_implicit(case: Dict[str, Any], cfg: Dict[str, Any]):
    dcfg = cfg.get("dialogue_test", {})
    if isinstance(case.get("goal"), dict):
        goal = case.get("goal", {}) or {}
        explicit_raw = goal.get("explicit_inform_slots", {}) or {}
        implicit_raw = goal.get("implicit_inform_slots", {}) or {}
    else:
        explicit_key = dcfg.get("initial_symptoms_key", "self_report")
        all_key = dcfg.get("all_symptoms_key", "all_symptoms")
        explicit_raw = case.get(explicit_key, {}) or {}
        all_raw = case.get(all_key, {}) or {}
        if not isinstance(explicit_raw, dict):
            raise TypeError(f"case[{explicit_key!r}] must be dict")
        if not isinstance(all_raw, dict):
            raise TypeError(f"case[{all_key!r}] must be dict")
        implicit_raw = {k: v for k, v in all_raw.items() if k not in explicit_raw}
    explicit = {str(k): normalize_case_value(v) for k, v in explicit_raw.items()}
    implicit = {str(k): normalize_case_value(v) for k, v in implicit_raw.items()}
    return explicit, implicit


def build_invalid_ids(vocab: Dict[str, int], mask_special_tokens: bool = True) -> set:
    invalid = set()
    if "[PAD]" in vocab:
        invalid.add(int(vocab["[PAD]"]))
    if mask_special_tokens:
        for sp in ["[UNK]", "[MASK]", "[PAD2]", "[CLS]", "[SEP]"]:
            if sp in vocab:
                invalid.add(int(vocab[sp]))
    return invalid


def load_diffdiag_vocab(data_path: str, vocab_file: str = ""):
    vocab_raw = None
    id2token_raw = None
    if data_path:
        data_path = os.path.expanduser(data_path)
        vocab_path = os.path.join(data_path, "token_vocab.json")
        id2token_path = os.path.join(data_path, "id2token.json")
        if os.path.exists(vocab_path):
            vocab_raw = load_json(vocab_path)
        if os.path.exists(id2token_path):
            id2token_raw = load_json(id2token_path)
    if vocab_raw is None:
        if not vocab_file:
            raise FileNotFoundError("No DiffDiag vocab found.")
        vocab_raw = load_json(os.path.expanduser(vocab_file))
    vocab = normalize_token_vocab(vocab_raw)
    if id2token_raw is not None:
        if all(str(k).isdigit() for k in id2token_raw.keys()):
            id2token = {int(k): str(v) for k, v in id2token_raw.items()}
        else:
            id2token = invert_token_vocab(normalize_token_vocab(id2token_raw))
    else:
        id2token = invert_token_vocab(vocab)
    return vocab, id2token


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_diffusion_pack(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    dcfg = cfg.get("dialogue_test", {})
    _add_diffdiag_path(dcfg["diffdiag_code_dir"])
    try:
        import models.gaussian_diffusion as gd
    except Exception as e:
        raise ImportError(f"Cannot import DiffDiag models.gaussian_diffusion: {e}") from e

    vocab, id2token = load_diffdiag_vocab(
        dcfg.get("diffusion_data_path", ""),
        dcfg.get("diffusion_vocab_file", ""),
    )

    diffusion_ckpt = os.path.expanduser(dcfg["diffusion_ckpt"])
    diffusion_ae_ckpt = os.path.expanduser(dcfg["diffusion_ae_ckpt"])
    print(f"[Diffusion] load model: {diffusion_ckpt}")
    model = safe_torch_load(diffusion_ckpt, device).to(device)
    print(f"[Diffusion] load AE: {diffusion_ae_ckpt}")
    ae = safe_torch_load(diffusion_ae_ckpt, device).to(device)
    model.eval()
    ae.eval()

    mean_type_name = str(dcfg.get("mean_type", "x0")).lower()
    if mean_type_name == "x0":
        mean_type = gd.ModelMeanType.START_X
    elif mean_type_name in ["eps", "epsilon"]:
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError(f"Unsupported mean_type: {mean_type_name}")

    steps = int(dcfg.get("steps", 5))
    sampling_steps = int(dcfg.get("sampling_steps", 0))
    if sampling_steps > steps:
        raise ValueError(f"sampling_steps={sampling_steps} > steps={steps}")

    diffusion = gd.GaussianDiffusion(
        mean_type,
        dcfg.get("noise_schedule", "linear-var"),
        float(dcfg.get("noise_scale", 0.1)),
        float(dcfg.get("noise_min", 0.0001)),
        float(dcfg.get("noise_max", 0.02)),
        steps,
        device,
    ).to(device)

    n_item = int(getattr(ae, "n_item", len(vocab)))
    if n_item != len(vocab):
        print(f"[Warning] AE.n_item={n_item}, vocab_size={len(vocab)}")

    pack = {
        "model": model, "AE": ae, "diffusion": diffusion,
        "vocab": vocab, "id2token": id2token,
        "invalid_ids": build_invalid_ids(vocab, bool(dcfg.get("mask_special_tokens", True))),
        "n_item": n_item,
        "n_cate": int(getattr(ae, "n_cate", 1)),
        "latent_dim": getattr(ae, "in_dims", None) and int(ae.in_dims[-1]) if hasattr(ae, "in_dims") and getattr(ae, "in_dims") else None,
        "steps": steps, "sampling_steps": sampling_steps,
    }
    print(f"[Diffusion] vocab_size={len(vocab)}, n_item={pack['n_item']}, "
          f"n_cate={pack['n_cate']}, steps={steps}, sampling_steps={sampling_steps}")
    return pack


def encode_symptoms_for_diffusion(symptoms: List[str], vocab: Dict[str, int],
                                   device: torch.device) -> torch.Tensor:
    x = torch.zeros(len(vocab), dtype=torch.float32, device=device)
    for s in symptoms:
        if s in vocab:
            idx = int(vocab[s])
            if 0 <= idx < x.numel():
                x[idx] = 1.0
    return x.unsqueeze(0)


@torch.no_grad()
def generate_diffusion_scores(current_obs: torch.Tensor, diffusion_pack: Dict[str, Any],
                               device: torch.device, eval_samples: int,
                               sampling_noise: bool) -> torch.Tensor:
    model = diffusion_pack["model"]
    ae = diffusion_pack["AE"]
    diffusion = diffusion_pack["diffusion"]

    current_obs = current_obs.float().to(device)
    bsz = current_obs.size(0)
    sample_num = max(1, int(eval_samples))

    obs_category, obs_latent, _ = ae.Encode(current_obs)
    latent_repeated = obs_latent.repeat_interleave(sample_num, dim=0)
    category_repeated = obs_category.repeat_interleave(sample_num, dim=0)

    sampled_latent = diffusion.p_sample(
        model, latent_repeated, int(diffusion_pack["sampling_steps"]),
        bool(sampling_noise), cond=latent_repeated,
    )

    pred_category = ae.Decode(sampled_latent).clone()
    pred_category[category_repeated.nonzero(as_tuple=True)] = -1e9

    if int(diffusion_pack.get("n_cate", 1)) > 1:
        category_map = ae.category_map.to(device).long()
        pred_output = torch.full(
            (pred_category.size(0), int(diffusion_pack.get("n_item", len(diffusion_pack["vocab"])))),
            -1e9, dtype=pred_category.dtype, device=device,
        )
        pred_output[:, category_map] = pred_category
    else:
        pred_output = pred_category

    scores = pred_output.view(bsz, sample_num, -1).mean(dim=1)
    for idx in diffusion_pack["invalid_ids"]:
        if 0 <= int(idx) < scores.size(-1):
            scores[:, int(idx)] = -1e9
    return scores


@torch.no_grad()
def predict_next_symptom_with_diffusion(
    current_symptoms: List[str], forbidden_symptoms: set,
    diffusion_pack: Dict[str, Any], cfg: Dict[str, Any], device: torch.device,
):
    dcfg = cfg.get("dialogue_test", {})
    current_obs = encode_symptoms_for_diffusion(current_symptoms, diffusion_pack["vocab"], device)
    scores = generate_diffusion_scores(
        current_obs=current_obs, diffusion_pack=diffusion_pack, device=device,
        eval_samples=int(dcfg.get("eval_samples", 1)),
        sampling_noise=bool(dcfg.get("sampling_noise", False)),
    )[0]

    filtered = scores.clone()
    if bool(dcfg.get("mask_current_tokens_eval", True)):
        current_ids = current_obs.nonzero(as_tuple=True)[1].detach().cpu().tolist()
        for idx in current_ids:
            if 0 <= int(idx) < filtered.numel():
                filtered[int(idx)] = -1e9

    forbid_same = bool(dcfg.get("forbid_same_symptom", True))
    for idx in range(filtered.numel()):
        tok = diffusion_pack["id2token"].get(int(idx))
        if tok is None:
            filtered[idx] = -1e9
            continue
        if forbid_same and tok in forbidden_symptoms:
            filtered[idx] = -1e9

    valid = filtered > -1e8
    if not bool(valid.any().item()):
        return None, []

    probs = torch.softmax(filtered, dim=-1)
    topk = min(int(dcfg.get("save_topk", 10)), int(valid.sum().item()))
    top_scores, top_indices = torch.topk(filtered, k=topk)
    ranked = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        tok = diffusion_pack["id2token"].get(int(idx))
        if tok is None:
            continue
        ranked.append({
            "symptom": tok, "token": tok,
            "score": float(score),
            "prob": float(probs[int(idx)].item()),
            "sigmoid_prob": float(torch.sigmoid(filtered[int(idx)]).item()),
        })
    return (ranked[0] if ranked else None), ranked


def should_take_second_symptom(top_tokens: List[Dict[str, Any]], cfg: Dict[str, Any]):
    dcfg = cfg.get("dialogue_test", {})
    mode = str(dcfg.get("ask_top2_mode", "off")).lower()
    if mode in ["off", "false", "0", "none"]:
        return False, "top2_disabled"
    if len(top_tokens) < 2:
        return False, "top2_not_available"
    if mode == "always":
        return True, "top2_always"
    if mode == "sigmoid_rel":
        first_s = float(top_tokens[0].get("sigmoid_prob", 0.0) or 0.0)
        second_s = float(top_tokens[1].get("sigmoid_prob", 0.0) or 0.0)
        relative = second_s / first_s if first_s > 0 else 0.0
        ok = (second_s >= float(dcfg.get("second_symptom_sigmoid_threshold", 0.90))
              and relative >= float(dcfg.get("second_symptom_relative_sigmoid_threshold", 0.90)))
        return ok, f"second_sigmoid={second_s:.6f}, first_sigmoid={first_s:.6f}, relative={relative:.6f}"
    raise ValueError(f"Unsupported ask_top2_mode={mode!r}")


def update_dialogue_state_with_symptom(
    symptom: str, implicit_slots: Dict[str, bool],
    current_slots: Dict[str, bool], diffusion_symptoms: List[str],
    used_symptoms: set, pred_state_set: set,
    append_unknown_to_diffusion: bool = False,
):
    used_symptoms.add(symptom)
    if symptom in implicit_slots:
        value = bool(implicit_slots[symptom])
        current_slots[symptom] = value
        diffusion_symptoms.append(symptom)
        state_token = make_state_token(symptom, value)
        pred_state_set.add(state_token)
        return ("true" if value else "false"), value, state_token, True
    else:
        if append_unknown_to_diffusion:
            diffusion_symptoms.append(symptom)
        return "unknown", None, f"{symptom}__unknown", False


def build_probe_row_from_slots(current_slots: Dict[str, bool], cfg: Dict[str, Any]):
    disease_names = list(cfg.get("_disease_names", []))
    row = {
        "source_type": "dialogue_state",
        "dataset_name": cfg.get("paths", {}).get("dataset_name", "DATA"),
        "candidate_diseases": disease_names,
        "current_turn_True": [s for s, v in current_slots.items() if bool(v)],
        "current_turn_False": [s for s, v in current_slots.items() if not bool(v)],
        "last_turn_True": [],
        "last_turn_False": [],
    }
    row["prompt_messages"] = build_turn_prompt_messages(row, cfg)
    return row


@torch.no_grad()
def predict_slots_with_hidden_probe(
    current_slots: Dict[str, bool], extractor: HiddenExtractor, pca,
    model: nn.Module, cfg: Dict[str, Any], id2label: Dict[int, str],
    device: torch.device, temperature: float, top_k: int,
):
    row = build_probe_row_from_slots(current_slots, cfg)
    feat = extractor.encode_rows([row])
    if pca is not None:
        feat = pca.transform(feat).astype(np.float32)
    xb = torch.tensor(feat, dtype=torch.float32, device=device)
    logits = model(xb) / max(float(temperature), 1e-6)
    prob = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
    pred_id = int(np.argmax(prob))
    top_items = make_topk_items(prob, id2label, top_k)
    return id2label[pred_id], float(prob[pred_id]), top_items, prob


# ═══════════════════════════════════════════════════════════════
# Dialogue prediction (main entry)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Dialogue metrics
# ═══════════════════════════════════════════════════════════════

def compute_dialogue_disease_metrics(gold_labels: List[str], pred_labels: List[str],
                                      label2id: Dict[str, int]) -> Dict[str, Any]:
    valid = [(g, p) for g, p in zip(gold_labels, pred_labels) if g in label2id and p in label2id]
    if not valid:
        return {"num_cases": 0, "accuracy": 0.0, "macro_f1": 0.0,
                "macro_precision": 0.0, "macro_recall": 0.0}
    y_true = np.array([label2id[g] for g, _ in valid], dtype=np.int64)
    y_pred = np.array([label2id[p] for _, p in valid], dtype=np.int64)
    out = {"num_cases": int(len(valid)), "accuracy": float((y_true == y_pred).mean())}
    if SKLEARN_OK:
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        out["macro_precision"] = float(p)
        out["macro_recall"] = float(r)
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    else:
        out["macro_f1"] = None
        out["macro_precision"] = None
        out["macro_recall"] = None
    return out


def compute_state_set_metrics(gold_sets: List[set], pred_sets: List[set]) -> Dict[str, Any]:
    sx_correct = 0
    sx_total = 0
    labels = sorted(set().union(*gold_sets, *pred_sets)) if gold_sets or pred_sets else []
    macro_p = macro_r = macro_f1 = 0.0

    for gold, pred in zip(gold_sets, pred_sets):
        sx_correct += len(gold & pred)
        sx_total += len(gold)

    for label in labels:
        tp = sum(1 for g, p in zip(gold_sets, pred_sets) if label in g and label in p)
        fp = sum(1 for g, p in zip(gold_sets, pred_sets) if label not in g and label in p)
        fn = sum(1 for g, p in zip(gold_sets, pred_sets) if label in g and label not in p)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1v = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        macro_p += prec
        macro_r += rec
        macro_f1 += f1v

    if labels:
        macro_p /= len(labels)
        macro_r /= len(labels)
        macro_f1 /= len(labels)

    return {
        "sx_rec": sx_correct / sx_total if sx_total > 0 else 0.0,
        "sx_rec_numerator": int(sx_correct),
        "sx_rec_denominator": int(sx_total),
        "implicit_state_micro_f1": (2 * macro_p * macro_r / (macro_p + macro_r)
                                     if macro_p + macro_r > 0 else 0.0),
        "implicit_state_macro_f1": macro_f1,
    }


def run_dialogue_predict(cfg: Dict[str, Any]) -> None:
    dcfg = cfg.get("dialogue_test", {})
    probe_dir = cfg["paths"]["probe_dir"]

    # Get label mapping from checkpoint, independent of feature_dir/label_map.json
    device = torch.device(dcfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model, ckpt = load_probe(os.path.join(probe_dir, "best_disease_probe.pt"), device)
    label2id = {str(k): int(v) for k, v in ckpt["label2id"].items()}
    id2label = {int(k): str(v) for k, v in ckpt["id2label"].items()}
    disease_names = ckpt.get("disease_names") or [id2label[i] for i in range(len(id2label))]
    cfg["_disease_names"] = disease_names

    test_cases_file = dcfg.get("test_cases_file") or cfg.get("paths", {}).get("test_data_file")
    if not test_cases_file:
        raise ValueError("dialogue_test requires test_cases_file or paths.test_data_file")
    cases = read_turn_file(test_cases_file)
    max_cases = int(dcfg.get("max_test_samples", cfg.get("test", {}).get("max_test_samples", -1)))
    if max_cases > 0:
        cases = cases[:max_cases]
    print(f"[Dialogue predict] Loaded cases: {len(cases)} from {test_cases_file}")

    diffusion_device = torch.device(
        dcfg.get("diffusion_device", str(device)) if torch.cuda.is_available() else "cpu")

    pca = load_saved_pca_object(probe_dir)
    extractor = HiddenExtractor(cfg)
    diffusion_pack = load_diffusion_pack(cfg, diffusion_device)

    thresholds = dcfg.get("thresholds")
    if thresholds is None:
        thresholds = [float(dcfg.get("threshold", 0.95))]
    thresholds = sorted(set(float(x) for x in thresholds))
    max_turns = int(dcfg.get("max_turns", 14))
    top_k = int(dcfg.get("top_k", cfg.get("test", {}).get("top_k", 5)))
    temperature = float(dcfg.get("temperature", cfg.get("test", {}).get("temperature",
                         cfg.get("probe", {}).get("temperature", 1.0))))
    append_unknown = bool(dcfg.get("append_unknown_to_diffusion", False))
    stop_when_all_gold = bool(dcfg.get("stop_when_all_gold_implicit_found", False))

    out_dir = dcfg.get("out_dir") or os.path.join(probe_dir, "dialogue_predict")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_predictions = []
    t_start = time.time()

    for case_idx, case in enumerate(tqdm(cases, desc="dialogue predict", dynamic_ncols=True)):
        case_id = get_case_id(case, case_idx)
        gold_label = get_case_label(case)
        explicit_slots, implicit_slots = get_case_explicit_implicit(case, cfg)

        current_slots = dict(explicit_slots)
        diffusion_symptoms = list(explicit_slots.keys())
        used_symptoms = set(explicit_slots.keys())
        implicit_symptoms_set = set(implicit_slots.keys())

        gold_state_set = {make_state_token(s, v) for s, v in implicit_slots.items()}
        pred_state_set = set()
        generated_symptoms = []
        simplified_turns = []
        generated_turns = 0

        final_pred = None
        final_prob = None

        while True:
            pred, pred_prob, _, _ = predict_slots_with_hidden_probe(
                current_slots=current_slots, extractor=extractor, pca=pca,
                model=model, cfg=cfg, id2label=id2label,
                device=device, temperature=temperature, top_k=top_k,
            )
            final_pred, final_prob = pred, pred_prob

            if pred_prob >= max(thresholds):
                break
            if generated_turns >= max_turns:
                break
            if stop_when_all_gold and implicit_symptoms_set and implicit_symptoms_set.issubset(set(generated_symptoms)):
                break

            next_item, top_tokens = predict_next_symptom_with_diffusion(
                current_symptoms=diffusion_symptoms, forbidden_symptoms=used_symptoms,
                diffusion_pack=diffusion_pack, cfg=cfg, device=diffusion_device,
            )
            if next_item is None:
                break

            take_second, _ = should_take_second_symptom(top_tokens, cfg)
            ask_items = list(top_tokens[:1])
            if take_second and len(top_tokens) >= 2:
                ask_items.append(top_tokens[1])

            for _, item in enumerate(ask_items, start=1):
                symptom = str(item["symptom"])
                if symptom in used_symptoms:
                    continue

                _, value, _, _ = update_dialogue_state_with_symptom(
                    symptom=symptom, implicit_slots=implicit_slots,
                    current_slots=current_slots, diffusion_symptoms=diffusion_symptoms,
                    used_symptoms=used_symptoms, pred_state_set=pred_state_set,
                    append_unknown_to_diffusion=append_unknown,
                )
                generated_symptoms.append(symptom)
                generated_turns += 1

                known_pos = [s for s, v in current_slots.items() if v and s not in implicit_slots]
                known_neg = [s for s, v in current_slots.items() if not v and s not in implicit_slots]
                simplified_turns.append({
                    "turn": generated_turns,
                    "known_positive": known_pos,
                    "known_negative": known_neg,
                    "asked": symptom,
                    "answer": bool(value) if value is not None else None,
                })

        case_record = {
            "id": case_id,
            "gold_disease": gold_label,
            "pred_disease": final_pred,
            "pred_prob": float(final_prob) if final_prob is not None else 0.0,
            "num_turns": generated_turns,
            "turns": simplified_turns,
            "_gold_state_set": sorted(gold_state_set),
            "_pred_state_set": sorted(pred_state_set),
        }
        all_predictions.append(case_record)

    t_total = time.time() - t_start

    # Save predictions
    save_json(os.path.join(out_dir, "dialogue_predictions.json"), all_predictions)

    # ── Compute metrics ──
    gold_labels = [r["gold_disease"] for r in all_predictions]
    pred_labels = [r["pred_disease"] for r in all_predictions]
    gold_state_sets = [set(r.get("_gold_state_set", [])) for r in all_predictions]
    pred_state_sets = [set(r.get("_pred_state_set", [])) for r in all_predictions]

    disease_m = compute_dialogue_disease_metrics(gold_labels, pred_labels, label2id)
    state_m = compute_state_set_metrics(gold_state_sets, pred_state_sets)

    n = len(all_predictions)
    total_turns = sum(r["num_turns"] for r in all_predictions)

    metrics = {
        "num_cases": n,
        "disease_accuracy": round(disease_m.get("accuracy", 0.0), 4),
        "sx_rec": round(state_m.get("sx_rec", 0.0), 4),
        "avg_turns": round(total_turns / n, 2) if n else 0.0,
        "total_time_s": round(t_total, 1),
        "time_per_case_s": round(t_total / n, 2) if n else 0.0,
    }
    save_json(os.path.join(out_dir, "dialogue_metrics.json"), metrics)

    print("[Dialogue metrics]")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_dir}")


# ═══════════════════════════════════════════════════════════════
# Single case prediction (for ad-hoc use)
# ═══════════════════════════════════════════════════════════════

def predict_single_case(
    cfg: Dict[str, Any],
    positive_symptoms: List[str],
    negative_symptoms: List[str],
    candidate_diseases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Predict disease for a single case given positive/negative symptoms.
    Loads probe, PCA, and LLM once — callable from other scripts.
    """
    _add_diffdiag_path(cfg.get("dialogue_test", {}).get("diffdiag_code_dir", ""))

    probe_dir = cfg["paths"]["probe_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_probe(os.path.join(probe_dir, "best_disease_probe.pt"), device)
    id2label = {int(k): str(v) for k, v in ckpt["id2label"].items()}
    disease_names = ckpt.get("disease_names") or [id2label[i] for i in range(len(id2label))]
    cfg = dict(cfg)
    cfg["_disease_names"] = disease_names

    pca = load_saved_pca_object(probe_dir)
    extractor = HiddenExtractor(cfg)

    pcfg = cfg.get("probe", {})
    dcfg = cfg.get("dialogue_test", {})
    top_k = int(dcfg.get("top_k", cfg.get("test", {}).get("top_k", 5)))
    temperature = float(dcfg.get("temperature", cfg.get("test", {}).get("temperature",
                         pcfg.get("temperature", 1.0))))

    row = {
        "source_type": "manual",
        "dataset_name": cfg.get("paths", {}).get("dataset_name", "DATA"),
        "candidate_diseases": candidate_diseases or disease_names,
        "current_turn_True": positive_symptoms,
        "current_turn_False": negative_symptoms,
        "last_turn_True": [],
        "last_turn_False": [],
    }
    row["prompt_messages"] = build_turn_prompt_messages(row, cfg)

    feat = extractor.encode_rows([row])
    if pca is not None:
        feat = pca.transform(feat).astype(np.float32)

    probs = predict_disease(model, feat, int(pcfg.get("batch_size", 128)), device, temperature)
    prob = probs[0]
    pred_id = int(np.argmax(prob))
    top_items = make_topk_items(prob, id2label, top_k)

    return {
        "pred_disease": id2label[pred_id],
        "pred_probability": float(prob[pred_id]),
        "topk": top_items,
        "disease_probabilities": {id2label[j]: float(prob[j]) for j in range(len(prob))},
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Standalone disease probe prediction")
    parser.add_argument("--config", type=str, required=True, help="JSON config file path")
    parser.add_argument("--mode", type=str, default="static",
                        choices=["static", "dialogue", "single"],
                        help="static: predict on pre-extracted features; "
                             "dialogue: interactive diffusion-based prediction; "
                             "single: predict one case from CLI args")
    # single-case args
    parser.add_argument("--pos", type=str, nargs="*", default=None,
                        help="Positive symptoms (for --mode single)")
    parser.add_argument("--neg", type=str, nargs="*", default=None,
                        help="Negative symptoms (for --mode single)")
    args = parser.parse_args()

    cfg = load_json(args.config)

    if args.mode == "static":
        run_static_predict(cfg)
    elif args.mode == "dialogue":
        run_dialogue_predict(cfg)
    elif args.mode == "single":
        if not cfg.get("dialogue_test", {}).get("diffdiag_code_dir"):
            # single mode doesn't need diffusion, provide a dummy
            cfg.setdefault("dialogue_test", {})["diffdiag_code_dir"] = "/tmp"
        result = predict_single_case(cfg, args.pos or [], args.neg or [])
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
