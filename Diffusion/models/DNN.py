import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


class DNN(nn.Module):
    """DNN for the reverse process of latent diffusion.

    Supports an optional condition vector. For symptom completion, cond is
    usually the clean latent vector of the observed / explicit symptom set.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False,
                 act_func='tanh', dropout=0.5, cond_dim=0):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_emb_dim = emb_size
        self.time_type = time_type
        self.norm = norm
        self.cond_dim = int(cond_dim)

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.cond_dim + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

        self.in_modules = []
        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
            self.in_modules.append(nn.Linear(d_in, d_out))
            self.in_modules.append(_activation(act_func))
        self.in_layers = nn.Sequential(*self.in_modules)

        self.out_modules = []
        for d_in, d_out in zip(out_dims[:-1], out_dims[1:]):
            self.out_modules.append(nn.Linear(d_in, d_out))
            self.out_modules.append(_activation(act_func))
        if len(self.out_modules) > 0:
            self.out_modules.pop()
        self.out_layers = nn.Sequential(*self.out_modules)

        self.dropout = nn.Dropout(dropout)
        self.apply(xavier_normal_initialization)

    def forward(self, x, timesteps, cond=None):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.dropout(x)

        if cond is not None:
            if self.norm:
                cond = F.normalize(cond, dim=-1)
            h = torch.cat([x, cond, emb], dim=-1)
        else:
            h = torch.cat([x, emb], dim=-1)

        h = self.in_layers(h)
        h = self.out_layers(h)
        return h


def _activation(act_func):
    if act_func == 'tanh':
        return nn.Tanh()
    if act_func == 'relu':
        return nn.ReLU()
    if act_func == 'sigmoid':
        return nn.Sigmoid()
    if act_func == 'leaky_relu':
        return nn.LeakyReLU()
    raise ValueError("Unsupported activation function: %s" % act_func)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def xavier_normal_initialization(module):
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
