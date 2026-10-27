import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class SymptomDiffusionDataset(Dataset):
    """Dataset for conditional symptom generation.

    Each sample contains:
      obs    : known / queried symptom vector, 0/1, shape [n_symptom]
      state  : symptom status prefix, e.g. -1/0/1, shape [n_symptom]
      full   : complete symptom vector, 0/1, shape [n_symptom]
      target : remaining target symptoms to generate/recommend, 0/1, shape [n_symptom]
      weight : item-level loss weights, shape [n_symptom]

    Important:
      - obs / full / target are symptom vectors and do not contain status values.
      - state is only used as a condition prefix. It is not a diffusion target.
      - In main.py, the condition is CondEncoder([state; obs]).
    """

    def __init__(self, obs, full, target, weight=None, state=None):
        self.obs = torch.FloatTensor(obs)
        self.full = torch.FloatTensor(full)
        self.target = torch.FloatTensor(target)

        if weight is None:
            weight = np.ones_like(full, dtype=np.float32)
        self.weight = torch.FloatTensor(weight)

        # If train_state.npy / test_state.npy does not exist, fall back to obs.
        # This keeps old 0/1 datasets runnable. For the real status prefix,
        # provide state arrays with values such as 1 positive, -1 negative, 0 unknown.
        if state is None:
            state = obs
        self.state = torch.FloatTensor(state)

        assert self.obs.shape == self.full.shape == self.target.shape == self.weight.shape
        assert self.state.shape == self.obs.shape

    def __getitem__(self, index):
        return {
            "obs": self.obs[index],
            "state": self.state[index],
            "full": self.full[index],
            "target": self.target[index],
            "weight": self.weight[index],
        }

    def __len__(self):
        return self.obs.shape[0]


# Keep original Dataset name for compatibility with old imports.
class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def _load_required_array(data_path, split, name):
    path = os.path.join(data_path, f"{split}_{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Please run build_symptom_dataset.py first.")
    return np.load(path, allow_pickle=True)


def _load_optional_array(data_path, split, name, default=None):
    path = os.path.join(data_path, f"{split}_{name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return default


def symptom_data_load(data_path):
    """Load train/test arrays produced by the symptom dataset builder.

    Required files:
      train_obs.npy, train_full.npy, train_target.npy, train_weight.npy
      test_obs.npy,  test_full.npy,  test_target.npy,  test_weight.npy

    Optional files:
      train_state.npy, test_state.npy

    The optional state arrays are the status prefix. They should have the same
    shape as obs and target, but they are not symptom targets. A common coding is:
      1  = positive symptom
     -1  = negative symptom
      0  = unknown / not observed
    """
    train_obs = _load_required_array(data_path, "train", "obs")
    train_full = _load_required_array(data_path, "train", "full")
    train_target = _load_required_array(data_path, "train", "target")
    train_weight = _load_required_array(data_path, "train", "weight")
    train_state = _load_optional_array(data_path, "train", "state", default=train_obs)

    test_obs = _load_required_array(data_path, "test", "obs")
    test_full = _load_required_array(data_path, "test", "full")
    test_target = _load_required_array(data_path, "test", "target")
    test_weight = _load_required_array(data_path, "test", "weight")
    test_state = _load_optional_array(data_path, "test", "state", default=test_obs)

    n_item = train_obs.shape[1]
    assert train_full.shape[1] == n_item
    assert train_target.shape[1] == n_item
    assert train_weight.shape[1] == n_item
    assert train_state.shape[1] == n_item
    assert test_obs.shape[1] == n_item
    assert test_full.shape[1] == n_item
    assert test_target.shape[1] == n_item
    assert test_weight.shape[1] == n_item
    assert test_state.shape[1] == n_item

    train_dataset = SymptomDiffusionDataset(
        train_obs,
        train_full,
        train_target,
        train_weight,
        train_state,
    )
    test_dataset = SymptomDiffusionDataset(
        test_obs,
        test_full,
        test_target,
        test_weight,
        test_state,
    )

    print(f"train samples before runtime valid split: {len(train_dataset)}")
    print(f"test samples: {len(test_dataset)}")
    print(f"symptom/item num: {n_item}")
    if _load_optional_array(data_path, "train", "state", default=None) is None:
        print("train_state.npy not found; using train_obs as state fallback.")
    else:
        print("train_state.npy loaded as condition prefix.")
    if _load_optional_array(data_path, "test", "state", default=None) is None:
        print("test_state.npy not found; using test_obs as state fallback.")
    else:
        print("test_state.npy loaded as condition prefix.")

    return train_dataset, test_dataset, n_item


def split_train_valid_dataset(train_dataset, valid_size=0, valid_ratio=0.1, seed=42):
    """Split training samples into train/valid inside training code.

    valid_size has priority when > 0. Otherwise valid_ratio is used.
    """
    n = len(train_dataset)
    if n <= 1:
        return train_dataset, train_dataset

    if valid_size and valid_size > 0:
        n_valid = int(valid_size)
    else:
        n_valid = int(round(n * float(valid_ratio)))

    n_valid = max(1, min(n_valid, n - 1))
    n_train = n - n_valid

    generator = torch.Generator().manual_seed(seed)
    train_subset, valid_subset = random_split(train_dataset, [n_train, n_valid], generator=generator)
    print(f"runtime train samples: {len(train_subset)}")
    print(f"runtime valid samples: {len(valid_subset)}")
    return train_subset, valid_subset


def load_vocab(data_path):
    vocab_path = os.path.join(data_path, "token_vocab.json")
    id2token_path = os.path.join(data_path, "id2token.json")
    vocab, id2token = None, None
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    if os.path.exists(id2token_path):
        with open(id2token_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        id2token = {int(k): v for k, v in raw.items()}
    return vocab, id2token
