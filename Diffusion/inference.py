"""Inference for conditional diffusion symptom target generation.

Usage examples:
  python inference.py --cuda --dataset CMD --data_path ../data/processed_data/diffusion_model_data/CMD \
    --model_path ../saved_models/CMD/CMD_log_best.pth \
    --ae_path ../saved_models/CMD/CMD_log_best_AE.pth \
    --cond_path ../saved_models/CMD/CMD_log_best_COND.pth \
    --symptoms "咳嗽,流涕" --topK 10

  python inference.py --cuda --dataset CMD --data_path ../data/processed_data/diffusion_model_data/CMD \
    --model_path ../saved_models/CMD/CMD_log_best.pth \
    --ae_path ../saved_models/CMD/CMD_log_best_AE.pth \
    --cond_path ../saved_models/CMD/CMD_log_best_COND.pth \
    --eval_test
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.gaussian_diffusion as gd
import data_utils
import evaluate_utils


class ConditionEncoder(nn.Module):
    def __init__(self, n_item, latent_size, hidden_dim=None, dropout=0.1, act_func="tanh"):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_size

        if act_func == "relu":
            act = nn.ReLU()
        elif act_func == "sigmoid":
            act = nn.Sigmoid()
        elif act_func == "leaky_relu":
            act = nn.LeakyReLU()
        else:
            act = nn.Tanh()

        self.n_item = int(n_item)
        self.latent_size = int(latent_size)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.act_func = act_func

        self.net = nn.Sequential(
            nn.Linear(2 * self.n_item, self.hidden_dim),
            act,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_size),
        )

    def forward(self, state, obs):
        cond_input = torch.cat([state.float(), obs.float()], dim=-1)
        return self.net(cond_input)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def split_tokens(text):
    if text is None or not text.strip():
        return []
    return [x.strip() for x in text.replace('，', ',').split(',') if x.strip()]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CMD')
parser.add_argument('--data_path', type=str, default='../data/processed_data/diffusion_model_data/CMD')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--ae_path', type=str, required=True)
parser.add_argument('--cond_path', type=str, required=True)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--topK', type=int, default=10)
parser.add_argument('--topN', type=str, default='[1, 3, 5, 10]')
parser.add_argument('--symptoms', type=str, default='',
                    help='comma-separated observed symptoms; treated as positive if positive/negative lists are not provided')
parser.add_argument('--positive_symptoms', type=str, default='', help='comma-separated positive observed symptoms')
parser.add_argument('--negative_symptoms', type=str, default='', help='comma-separated negative observed symptoms')
parser.add_argument('--eval_test', action='store_true')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--mean_type', type=str, default='x0')
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--noise_schedule', type=str, default='linear-var')
parser.add_argument('--noise_scale', type=float, default=0.1)
parser.add_argument('--noise_min', type=float, default=0.0001)
parser.add_argument('--noise_max', type=float, default=0.02)
parser.add_argument('--sampling_noise', type=str2bool, default=False)
parser.add_argument('--sampling_steps', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

vocab, id2token = data_utils.load_vocab(args.data_path)
if vocab is None or id2token is None:
    raise FileNotFoundError("token_vocab.json / id2token.json not found in data_path")

model = torch.load(args.model_path, map_location=device, weights_only=False).to(device)
Autoencoder = torch.load(args.ae_path, map_location=device, weights_only=False).to(device)

cond_ckpt = torch.load(args.cond_path, map_location=device, weights_only=False)
CondEncoder = ConditionEncoder(
    n_item=cond_ckpt["n_item"],
    latent_size=cond_ckpt["latent_size"],
    hidden_dim=cond_ckpt["hidden_dim"],
    dropout=cond_ckpt["dropout"],
    act_func=cond_ckpt.get("act_func", "tanh"),
).to(device)
CondEncoder.load_state_dict(cond_ckpt["state_dict"])

model.eval()
Autoencoder.eval()
CondEncoder.eval()

if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError(args.mean_type)

diffusion = gd.GaussianDiffusion(
    mean_type,
    args.noise_schedule,
    args.noise_scale,
    args.noise_min,
    args.noise_max,
    args.steps,
    device,
).to(device)


def build_obs_and_state_from_cli():
    obs_vec = np.zeros(len(vocab), dtype=np.float32)
    state_vec = np.zeros(len(vocab), dtype=np.float32)
    unknown = []

    pos_tokens = split_tokens(args.positive_symptoms)
    neg_tokens = split_tokens(args.negative_symptoms)

    # Backward compatibility: --symptoms means observed positive symptoms if no
    # explicit positive/negative symptom list is given.
    if not pos_tokens and not neg_tokens:
        pos_tokens = split_tokens(args.symptoms)
    else:
        # If the user also passes --symptoms with explicit state lists, treat it
        # as extra observed symptoms with unknown/non-negative state 1 by default.
        pos_tokens = pos_tokens + split_tokens(args.symptoms)

    for token in pos_tokens:
        if token in vocab:
            idx = int(vocab[token])
            obs_vec[idx] = 1.0
            state_vec[idx] = 1.0
        else:
            unknown.append(token)

    for token in neg_tokens:
        if token in vocab:
            idx = int(vocab[token])
            obs_vec[idx] = 1.0
            state_vec[idx] = -1.0
        else:
            unknown.append(token)

    return obs_vec, state_vec, unknown


def predict_from_vector(obs_vec, state_vec, topK):
    obs = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
    state = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

    with torch.no_grad():
        obs_cate = Autoencoder.MapInput(obs)

        # Condition is encoded after concatenating [state; obs].
        cond_latent = CondEncoder(state, obs)

        # Reverse process starts from random noise and denoises to target latent.
        latent_recon = diffusion.p_sample_loop(
            model,
            cond_latent.shape,
            args.sampling_noise,
            cond=cond_latent,
        )
        prediction = Autoencoder.Decode(latent_recon)
        prediction[obs_cate.nonzero(as_tuple=True)] = -float('inf')

        k = min(topK, prediction.shape[1])
        values, mapped_indices = torch.topk(prediction, k)
        if Autoencoder.n_cate > 1:
            indices = Autoencoder.category_map.to(device)[mapped_indices]
        else:
            indices = mapped_indices

    results = []
    for idx, score in zip(indices[0].cpu().numpy().tolist(), values[0].cpu().numpy().tolist()):
        results.append((id2token[int(idx)], float(score)))
    return results


def evaluate_test():
    _, test_dataset, _ = data_utils.symptom_data_load(args.data_path)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    predict_items, target_items = [], []
    max_k = max(eval(args.topN))
    category_map = Autoencoder.category_map.to(device) if Autoencoder.n_cate > 1 else None

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            state = batch["state"].to(device)
            target = batch["target"].to(device)

            for row in target.cpu().numpy():
                target_items.append(np.nonzero(row)[0].tolist())

            obs_cate = Autoencoder.MapInput(obs)
            cond_latent = CondEncoder(state, obs)
            latent_recon = diffusion.p_sample_loop(
                model,
                cond_latent.shape,
                args.sampling_noise,
                cond=cond_latent,
            )
            prediction = Autoencoder.Decode(latent_recon)
            prediction[obs_cate.nonzero(as_tuple=True)] = -float('inf')

            _, mapped_indices = torch.topk(prediction, min(max_k, prediction.shape[1]))
            if category_map is not None:
                indices = category_map[mapped_indices]
            else:
                indices = mapped_indices
            predict_items.extend(indices.cpu().numpy().tolist())

    res = evaluate_utils.computeTopNAccuracy(target_items, predict_items, eval(args.topN))
    evaluate_utils.print_results(None, None, res)


if args.eval_test:
    evaluate_test()

if args.symptoms.strip() or args.positive_symptoms.strip() or args.negative_symptoms.strip():
    obs_vec, state_vec, unknown = build_obs_and_state_from_cli()
    if unknown:
        print("Unknown symptoms skipped:", unknown)
    results = predict_from_vector(obs_vec, state_vec, args.topK)
    print("Top hidden symptom candidates:")
    for rank, (symptom, score) in enumerate(results, start=1):
        print(f"{rank}. {symptom}\t{score:.6f}")
