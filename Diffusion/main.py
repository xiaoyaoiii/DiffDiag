"""Train conditional DiffDiag for symptom target generation.

Pipeline:
  target symptom vector -> AutoEncoder latent -> GaussianDiffusion + DNN -> decoded target scores.

This version follows the state-prefix design:
  - Symptom vectors obs / target are still 0/1 and do not contain status values.
  - The diffusion target is target_latent, i.e. noise is added only to the target symptoms.
  - The condition is encoded from the prefix [state; obs]:
        cond_latent = CondEncoder(torch.cat([state, obs], dim=-1))
  - state only guides denoising as a condition and is never used as the noisy target.
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models.gaussian_diffusion as gd
from models.Autoencoder import AutoEncoder as AE
from models.Autoencoder import compute_loss, weighted_bce_loss
from models.DNN import DNN
import evaluate_utils
import data_utils


random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class ConditionEncoder(nn.Module):
    """Encode the condition prefix [state; obs] into a latent condition vector.

    state: [B, n_item], e.g. 1 positive, -1 negative, 0 unknown.
    obs  : [B, n_item], 0/1 known or queried symptoms.

    The status prefix is not a generated target. It is concatenated with obs
    before encoding and then used only as cond for the denoising network.
    """

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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CMD', help='dataset name')
parser.add_argument('--data_path', type=str, default='../data/processed_data/diffusion_model_data/CMD', help='load data path')
parser.add_argument('--emb_path', type=str, default='../data/processed_data/diffusion_model_data', help='item embedding root path')
parser.add_argument('--lr1', type=float, default=0.0005, help='learning rate for Autoencoder')
parser.add_argument('--lr2', type=float, default=0.0001, help='learning rate for DNN and ConditionEncoder')
parser.add_argument('--wd1', type=float, default=0.0, help='weight decay for Autoencoder')
parser.add_argument('--wd2', type=float, default=0.0, help='weight decay for DNN and ConditionEncoder')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[1, 3, 5, 10]')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--valid_ratio', type=float, default=0.1, help='split this ratio from train samples as validation at training time')
parser.add_argument('--valid_size', type=int, default=0, help='if >0, split this many train samples as validation at training time')
parser.add_argument('--seed', type=int, default=42, help='seed for runtime train/valid split')

# params for Autoencoder
parser.add_argument('--n_cate', type=int, default=1, help='category num of symptoms/items; use 1 for small vocab')
parser.add_argument('--in_dims', type=str, default='[64]', help='encoder dims; final dim is latent size')
parser.add_argument('--out_dims', type=str, default='[]', help='decoder hidden dims')
parser.add_argument('--act_func', type=str, default='tanh', help='activation function for autoencoder')
parser.add_argument('--lamda', type=float, default=0.03, help='hyper-parameter for diffusion loss')
parser.add_argument('--optimizer1', type=str, default='AdamW', help='optimizer for AE')
parser.add_argument('--anneal_cap', type=float, default=0.005)
parser.add_argument('--anneal_steps', type=int, default=500)
parser.add_argument('--vae_anneal_cap', type=float, default=0.3)
parser.add_argument('--vae_anneal_steps', type=int, default=200)
parser.add_argument('--reparam', type=str2bool, default=True)
parser.add_argument('--hidden_loss_weight', type=float, default=1.0,
                    help='extra weighted BCE loss on target vector')

# params for DNN and condition encoder
parser.add_argument('--time_type', type=str, default='cat')
parser.add_argument('--mlp_dims', type=str, default='[64]')
parser.add_argument('--norm', type=str2bool, default=False)
parser.add_argument('--emb_size', type=int, default=10)
parser.add_argument('--mlp_act_func', type=str, default='tanh')
parser.add_argument('--optimizer2', type=str, default='AdamW')
parser.add_argument('--mlp_dropout', type=float, default=0.5)
parser.add_argument('--cond_hidden_dim', type=int, default=0,
                    help='hidden size of ConditionEncoder; 0 means latent_size')
parser.add_argument('--cond_dropout', type=float, default=0.1,
                    help='dropout of ConditionEncoder')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='x0 is recommended for conditional symptom completion')
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--noise_schedule', type=str, default='linear-var')
parser.add_argument('--noise_scale', type=float, default=0.1)
parser.add_argument('--noise_min', type=float, default=0.0001)
parser.add_argument('--noise_max', type=float, default=0.02)
parser.add_argument('--sampling_noise', type=str2bool, default=False)
parser.add_argument('--sampling_steps', type=int, default=0)
parser.add_argument('--reweight', type=str2bool, default=True)

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
print("device:", device)
print("Starting time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


# DATA LOAD
full_train_dataset, test_dataset, n_item = data_utils.symptom_data_load(args.data_path)
train_dataset, valid_dataset = data_utils.split_train_valid_dataset(
    full_train_dataset,
    valid_size=args.valid_size,
    valid_ratio=args.valid_ratio,
    seed=args.seed,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    pin_memory=args.cuda,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=worker_init_fn,
)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print('data ready.')


# Build Autoencoder for symptom vectors only. State is not encoded by this AE.
emb_path = os.path.join(args.emb_path, args.dataset, 'item_emb.npy')
if not os.path.exists(emb_path):
    emb_path = os.path.join(args.data_path, 'item_emb.npy')
item_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True)).float()
assert len(item_emb) == n_item, f"item_emb length {len(item_emb)} != n_item {n_item}"

out_dims = eval(args.out_dims)
in_dims = eval(args.in_dims)
Autoencoder = AE(item_emb, args.n_cate, in_dims, out_dims, device, args.act_func, args.reparam).to(device)


# Build Gaussian Diffusion
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(
    mean_type,
    args.noise_schedule,
    args.noise_scale,
    args.noise_min,
    args.noise_max,
    args.steps,
    device,
).to(device)


# Build condition encoder and denoising DNN.
# DNN cond_dim is latent_size because CondEncoder([state; obs]) outputs latent_size.
latent_size = in_dims[-1]
cond_hidden_dim = args.cond_hidden_dim if args.cond_hidden_dim > 0 else latent_size

CondEncoder = ConditionEncoder(
    n_item=n_item,
    latent_size=latent_size,
    hidden_dim=cond_hidden_dim,
    dropout=args.cond_dropout,
    act_func=args.mlp_act_func,
).to(device)

mlp_out_dims = eval(args.mlp_dims) + [latent_size]
mlp_in_dims = mlp_out_dims[::-1]
model = DNN(
    mlp_in_dims,
    mlp_out_dims,
    args.emb_size,
    time_type=args.time_type,
    norm=args.norm,
    act_func=args.mlp_act_func,
    dropout=args.mlp_dropout,
    cond_dim=latent_size,
).to(device)

AE_num = sum([param.nelement() for param in Autoencoder.parameters()])
cond_num = sum([param.nelement() for param in CondEncoder.parameters()])
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])
print("Number of parameters:", AE_num + cond_num + mlp_num + diff_num)
print(f"ConditionEncoder: input_dim={2 * n_item}, hidden_dim={cond_hidden_dim}, output_dim={latent_size}")


def build_optimizer(name, params, lr, wd):
    if name == 'Adagrad':
        return optim.Adagrad(params, lr=lr, initial_accumulator_value=1e-8, weight_decay=wd)
    if name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == 'SGD':
        return optim.SGD(params, lr=lr, weight_decay=wd)
    if name == 'Momentum':
        return optim.SGD(params, lr=lr, momentum=0.95, weight_decay=wd)
    raise ValueError("Unknown optimizer: %s" % name)


optimizer1 = build_optimizer(args.optimizer1, Autoencoder.parameters(), args.lr1, args.wd1)
optimizer2 = build_optimizer(
    args.optimizer2,
    list(model.parameters()) + list(CondEncoder.parameters()),
    args.lr2,
    args.wd2,
)
print("models ready.")


def evaluate(data_loader, topN):
    model.eval()
    Autoencoder.eval()
    CondEncoder.eval()

    predict_items = []
    target_items = []
    max_k = max(topN)
    category_map = Autoencoder.category_map.to(device) if args.n_cate > 1 else None

    with torch.no_grad():
        for batch in data_loader:
            obs = batch["obs"].to(device)          # 0/1 known symptoms
            state = batch["state"].to(device)      # status prefix
            target = batch["target"].to(device)    # 0/1 target symptoms

            for row in target.cpu().numpy():
                target_items.append(np.nonzero(row)[0].tolist())

            # Masking uses category-order symptom vector, not encoded latent.
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

            # Do not recommend already known symptoms.
            prediction[obs_cate.nonzero(as_tuple=True)] = -float('inf')
            k = min(max_k, prediction.shape[1])
            _, mapped_indices = torch.topk(prediction, k)
            if args.n_cate > 1:
                indices = category_map[mapped_indices]
            else:
                indices = mapped_indices
            predict_items.extend(indices.cpu().numpy().tolist())

    return evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)


best_recall, best_epoch = -100.0, 0
best_results, best_test_results = None, None
update_count = 0
update_count_vae = 0
save_path = os.path.join(args.save_path, args.dataset)
os.makedirs(save_path, exist_ok=True)

print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= args.patience:
        print('-' * 18)
        print('Exiting from training early')
        break

    Autoencoder.train()
    model.train()
    CondEncoder.train()

    start_time = time.time()
    total_loss = 0.0
    batch_count = 0

    for batch in train_loader:
        obs = batch["obs"].to(device)          # 0/1 known symptoms
        state = batch["state"].to(device)      # status prefix, not a target
        target = batch["target"].to(device)    # 0/1 target symptoms
        weight = batch["weight"].to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # Diffusion target: only target symptoms, no status values.
        target_cate, target_latent, vae_kl = Autoencoder.Encode(target)

        # Condition: concatenate state before obs, then encode the whole prefix.
        cond_latent = CondEncoder(state, obs)

        terms = diffusion.training_losses(
            model,
            target_latent,
            args.reweight,
            cond=cond_latent,
        )
        elbo = terms["loss"].mean()

        latent_recon = terms["pred_xstart"]
        target_recon = Autoencoder.Decode(latent_recon)

        if args.anneal_steps > 0:
            lamda = max((1.0 - update_count / args.anneal_steps) * args.lamda, args.anneal_cap)
        else:
            lamda = max(args.lamda, args.anneal_cap)

        if args.vae_anneal_steps > 0:
            anneal = min(args.vae_anneal_cap, 1.0 * update_count_vae / args.vae_anneal_steps)
        else:
            anneal = args.vae_anneal_cap

        # Reconstruction loss is also on the target symptom vector only.
        vae_loss = compute_loss(target_recon, target_cate) + anneal * vae_kl
        weight_cate = Autoencoder.MapInput(weight)
        bce_loss = weighted_bce_loss(target_recon, target_cate, weight_cate)
        vae_loss = vae_loss + args.hidden_loss_weight * bce_loss

        if args.reweight:
            loss = lamda * elbo + vae_loss
        else:
            loss = elbo + lamda * vae_loss

        loss.backward()
        optimizer1.step()
        optimizer2.step()

        update_count += 1
        update_count_vae += 1
        total_loss += float(loss.detach().cpu().item())
        batch_count += 1

    avg_loss = total_loss / max(batch_count, 1)

    if epoch % args.eval_interval == 0:
        topN = eval(args.topN)
        valid_results = evaluate(valid_loader, topN)
        test_results = evaluate(test_loader, topN)
        evaluate_utils.print_results(avg_loss, valid_results, test_results)

        # recall@topN[1] if exists, otherwise recall@topN[0]
        select_idx = 1 if len(topN) > 1 else 0
        if valid_results[1][select_idx] > best_recall:
            best_recall, best_epoch = valid_results[1][select_idx], epoch
            best_results = valid_results
            best_test_results = test_results

            model_file = os.path.join(save_path, f"{args.dataset}_{args.log_name}_best.pth")
            ae_file = os.path.join(save_path, f"{args.dataset}_{args.log_name}_best_AE.pth")
            cond_file = os.path.join(save_path, f"{args.dataset}_{args.log_name}_best_COND.pth")

            torch.save(model, model_file)
            torch.save(Autoencoder, ae_file)
            torch.save({
                "state_dict": CondEncoder.state_dict(),
                "n_item": n_item,
                "latent_size": latent_size,
                "hidden_dim": cond_hidden_dim,
                "dropout": args.cond_dropout,
                "act_func": args.mlp_act_func,
            }, cond_file)

            print(f"Saved best model to {model_file}")
            print(f"Saved best AE to {ae_file}")
            print(f"Saved best condition encoder to {cond_file}")

    print("Runing Epoch {:03d} train loss {:.4f} costs {}".format(
        epoch,
        avg_loss,
        time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)),
    ))

print('---' * 18)
print('===' * 18)
print("End. Best Epoch {:03d}".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)
print("End time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
