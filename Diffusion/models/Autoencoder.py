import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

try:
    from kmeans_pytorch import kmeans as torch_kmeans
except Exception:
    torch_kmeans = None


class AutoEncoder(nn.Module):
    """Autoencoder for symptom vectors.

    Exposes MapInput(), which maps an original symptom vector to the clustered
    category order used by Decode() when n_cate > 1.
    """

    def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func,
                 reparam=True, dropout=0.1):
        super(AutoEncoder, self).__init__()
        if isinstance(item_emb, np.ndarray):
            item_emb = torch.from_numpy(item_emb).float()
        self.item_emb = item_emb.float().cpu()
        self.n_cate = int(n_cate)
        self.in_dims = list(map(int, in_dims))
        self.out_dims = list(map(int, out_dims))
        self.act_func = act_func
        self.n_item = int(len(item_emb))
        self.reparam = reparam
        self.dropout = nn.Dropout(dropout)

        if self.n_cate == 1:
            in_dims_temp = [self.n_item] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
            out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item]

            encoder_modules = []
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                encoder_modules.append(nn.Linear(d_in, d_out))
                encoder_modules.append(_activation(self.act_func))
            self.encoder = nn.Sequential(*encoder_modules)

            decoder_modules = []
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
                decoder_modules.append(nn.Linear(d_in, d_out))
                decoder_modules.append(_activation(self.act_func))
            if len(decoder_modules) > 0:
                decoder_modules.pop()
            self.decoder = nn.Sequential(*decoder_modules)
            self.category_idx = None
            self.category_map = torch.arange(self.n_item, dtype=torch.long)
            self.category_len = [self.n_item]
            self.decode_dim = [self.in_dims[::-1]]
        else:
            self.cluster_ids = self._cluster_items(self.item_emb, self.n_cate, device)
            category_idx = []
            for i in range(self.n_cate):
                idx = np.argwhere(self.cluster_ids.numpy() == i).reshape(-1).tolist()
                if len(idx) == 0:
                    # Avoid empty category for small symptom vocabularies.
                    idx = [i % self.n_item]
                category_idx.append(torch.tensor(idx, dtype=torch.long))
            self.category_idx = category_idx
            self.category_map = torch.cat(tuple(category_idx), dim=-1).long()
            self.category_len = [len(self.category_idx[i]) for i in range(self.n_cate)]
            print("category length: ", self.category_len)

            encoder_modules = [[] for _ in range(self.n_cate)]
            decode_dim = []
            in_dims_np = np.array(self.in_dims, dtype=np.int64)
            for i in range(self.n_cate):
                if i == self.n_cate - 1:
                    used = np.array(decode_dim, dtype=np.int64).sum(axis=0) if len(decode_dim) > 0 else 0
                    latent_dims = list((in_dims_np - used).astype(int))
                else:
                    latent_dims = [int(self.category_len[i] / self.n_item * self.in_dims[j]) for j in range(len(self.in_dims))]
                latent_dims = [max(int(x), 1) for x in latent_dims]
                in_dims_temp = [self.category_len[i]] + latent_dims[:-1] + [latent_dims[-1] * 2]
                decode_dim.append(latent_dims)

                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                    encoder_modules[i].append(nn.Linear(d_in, d_out))
                    encoder_modules[i].append(_activation(self.act_func))

            self.encoder = nn.ModuleList([nn.Sequential(*encoder_modules[i]) for i in range(self.n_cate)])
            print("Latent dims of each category: ", decode_dim)
            self.decode_dim = [decode_dim[i][::-1] for i in range(len(decode_dim))]

            if len(self.out_dims) == 0:
                self.decoder = nn.Sequential(nn.Linear(self.in_dims[-1], self.n_item))
            else:
                decoder_modules = [[] for _ in range(self.n_cate)]
                for i in range(self.n_cate):
                    out_dims_temp = self.decode_dim[i] + [self.category_len[i]]
                    for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
                        decoder_modules[i].append(nn.Linear(d_in, d_out))
                        decoder_modules[i].append(_activation(self.act_func))
                    if len(decoder_modules[i]) > 0:
                        decoder_modules[i].pop()
                self.decoder = nn.ModuleList([nn.Sequential(*decoder_modules[i]) for i in range(self.n_cate)])

        self.apply(xavier_normal_initialization)

    def _cluster_items(self, item_emb, n_cate, device):
        if torch_kmeans is not None:
            cluster_ids, _ = torch_kmeans(X=item_emb, num_clusters=n_cate, distance='euclidean', device=device)
            return cluster_ids.cpu().long()
        return _simple_kmeans(item_emb, n_cate, seed=1)

    def MapInput(self, batch):
        if self.n_cate == 1:
            return batch
        parts = []
        for idx in self.category_idx:
            parts.append(batch[:, idx.to(batch.device)])
        return torch.cat(tuple(parts), dim=-1)

    def Encode(self, batch):
        batch = self.dropout(batch)
        if self.n_cate == 1:
            hidden = self.encoder(batch)
            mu = hidden[:, :self.in_dims[-1]]
            logvar = hidden[:, self.in_dims[-1]:]
            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            return batch, latent, kl_divergence

        batch_cate = []
        for i in range(self.n_cate):
            batch_cate.append(batch[:, self.category_idx[i].to(batch.device)])

        latent_mu = []
        latent_logvar = []
        for i in range(self.n_cate):
            hidden = self.encoder[i](batch_cate[i])
            dim = self.decode_dim[i][0]
            latent_mu.append(hidden[:, :dim])
            latent_logvar.append(hidden[:, dim:])

        mu = torch.cat(tuple(latent_mu), dim=-1)
        logvar = torch.cat(tuple(latent_logvar), dim=-1)
        if self.training and self.reparam:
            latent = self.reparamterization(mu, logvar)
        else:
            latent = mu
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return torch.cat(tuple(batch_cate), dim=-1), latent, kl_divergence

    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def Decode(self, batch):
        if len(self.out_dims) == 0 or self.n_cate == 1:
            return self.decoder(batch)

        batch_cate = []
        start = 0
        for i in range(self.n_cate):
            end = start + self.decode_dim[i][0]
            batch_cate.append(batch[:, start:end])
            start = end

        pred_cate = []
        for i in range(self.n_cate):
            pred_cate.append(self.decoder[i](batch_cate[i]))
        return torch.cat(tuple(pred_cate), dim=-1)


def compute_loss(recon_x, x):
    # multinomial log likelihood
    return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))


def weighted_bce_loss(logits, target, weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    denom = weight.sum(dim=1).clamp_min(1.0)
    return ((loss * weight).sum(dim=1) / denom).mean()


def _activation(act_func):
    if act_func == 'relu':
        return nn.ReLU()
    if act_func == 'sigmoid':
        return nn.Sigmoid()
    if act_func == 'tanh':
        return nn.Tanh()
    if act_func == 'leaky_relu':
        return nn.LeakyReLU()
    raise ValueError("Unsupported activation function: %s" % act_func)


def xavier_normal_initialization(module):
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def _simple_kmeans(X, k, seed=1, iters=50):
    """Small fallback k-means to avoid an extra dependency on tiny symptom vocabularies."""
    X = X.float().cpu()
    n = X.shape[0]
    rng = torch.Generator().manual_seed(seed)
    if n < k:
        k = n
    init = torch.randperm(n, generator=rng)[:k]
    centers = X[init].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        dist = torch.cdist(X, centers)
        new_labels = dist.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = X[mask].mean(dim=0)
    if k < int(labels.max().item()) + 1:
        pass
    return labels.long()
