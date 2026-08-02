"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".
"""

import numpy as np
import torch

randn_like = torch.randn_like

SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


def impute_mask(net, x, mask, num_samples, dim, num_steps=50, device="cuda:0"):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    x_t = torch.randn([num_samples, dim], device=device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    mask = mask.to(torch.int).to(device)
    x_t = x_t.to(torch.float32) * t_steps[0]

    N = 20
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            if i < num_steps - 1:
                for j in range(N):
                    _ = torch.randn_like(x_t).to(device) * t_cur
                    n_prev = torch.randn_like(x_t).to(device) * t_next

                    x_known_t_prev = x + n_prev
                    x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)

                    x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

                    n = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                    if j == N - 1:
                        x_t = x_t_prev
                    else:
                        x_t = x_t_prev + n
            else:
                _ = torch.randn_like(x_t).to(device) * t_cur
                n_prev = torch.randn_like(x_t).to(device) * t_next

                x_known_t_prev = x + n_prev
                x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)

                x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask
                x_t = x_t_prev

    return x_t


def sample(net, num_samples, dim, num_steps=50, device="cuda:0"):
    latents = torch.randn([num_samples, dim], device=device)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim=100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn, data):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)

        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)
        return loss
