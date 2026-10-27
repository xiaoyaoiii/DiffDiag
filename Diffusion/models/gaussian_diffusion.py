import enum
import math
import numpy as np
import torch as th
import torch.nn as nn


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,
                 steps, device, history_num_per_term=10, beta_fixed=False):
        super(GaussianDiffusion, self).__init__()
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=th.long).to(device)

        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"
            self.calculate_for_diffusion()

    def get_betas(self):
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        if self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        if self.noise_schedule == "binomial":
            ts = np.arange(self.steps)
            return np.array([1 / (self.steps - t + 1) for t in ts], dtype=np.float64)
        raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, model, x_start, steps, sampling_noise=False, cond=None):
        assert steps <= self.steps, "Too many steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        if self.noise_scale == 0.:
            for i in list(range(self.steps))[::-1]:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t, cond=cond)
            return x_t

        for i in list(range(self.steps))[::-1]:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t, cond=cond)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def p_sample_loop(self, model, shape, sampling_noise=False, cond=None):
        """Standard diffusion reverse process starting from pure Gaussian noise.

        x_T ~ N(0, I), then iteratively denoise x_T → x_{T-1} → ... → x_0,
        conditioned on cond.
        """
        batch_size = shape[0]
        device = self.device
        x_t = th.randn(*shape).to(device)

        if self.noise_scale == 0.:
            for i in list(range(self.steps))[::-1]:
                t = th.tensor([i] * batch_size).to(device)
                x_t = model(x_t, t, cond=cond)
            return x_t

        for i in list(range(self.steps))[::-1]:
            t = th.tensor([i] * batch_size).to(device)
            out = self.p_mean_variance(model, x_t, t, cond=cond)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def training_losses(self, model, x_start, reweight=False, target=None, cond=None):
        """Compute diffusion loss.

        x_start is the full-symptom latent (noise is added to it).
        cond is the observed-symptom latent.
        If target is None, target = x_start (standard diffusion denoising).
        """
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise) if self.noise_scale != 0. else x_start

        model_output = model(x_t, ts, cond=cond)
        if target is None:
            target = x_start

        if self.mean_type == ModelMeanType.START_X:
            diff_target = target
        elif self.mean_type == ModelMeanType.EPSILON:
            diff_target = noise
        else:
            raise NotImplementedError(self.mean_type)

        assert model_output.shape == diff_target.shape == x_start.shape
        mse = mean_flat((diff_target - model_output) ** 2)

        if reweight:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), th.ones_like(weight), weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts]))
                weight = th.where((ts == 0), th.ones_like(weight), weight)
                likelihood = mean_flat((target - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
            else:
                raise NotImplementedError(self.mean_type)
        else:
            weight = th.ones_like(mse).to(device)
            loss = mse

        terms = {}
        terms["loss"] = weight * loss
        if self.mean_type == ModelMeanType.START_X:
            terms["pred_xstart"] = model_output
        else:
            terms["pred_xstart"] = self._predict_xstart_from_eps(x_t, ts, model_output)

        # update Lt_history & Lt_count for importance sampling
        for t, one_loss in zip(ts, terms["loss"]):
            t_int = int(t.item())
            if self.Lt_count[t_int] == self.history_num_per_term:
                old = self.Lt_history.clone()
                self.Lt_history[t_int, :-1] = old[t_int, 1:]
                self.Lt_history[t_int, -1] = one_loss.detach()
            else:
                self.Lt_history[t_int, self.Lt_count[t_int]] = one_loss.detach()
                self.Lt_count[t_int] += 1

        terms["loss"] = terms["loss"] / pt
        return terms

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)
            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        if method == 'uniform':
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()
            return t, pt
        raise ValueError

    def q_sample(self, x_start, t, noise=None):
        if self.noise_scale == 0.:
            return x_start
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, cond=None):
        B = x.shape[0]
        assert t.shape == (B,)
        model_output = model(x, t, cond=cond)
        model_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t):
        t = th.clamp(t, min=0)
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = [1 - alpha_bar[0]]
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
