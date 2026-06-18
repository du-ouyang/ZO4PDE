"""Spiky conditional-expectation example.

This experiment mirrors the option-pricing diagnostic in the paper, but uses a
terminal payoff with many small narrow bumps.  The value function is

    u(t, x) = E[g(X_T) | X_t = x],

where X follows a one-dimensional arithmetic Brownian motion.  The Gaussian
transition smooths the payoff, but the value still contains small folded
features whose derivatives are much larger than their value amplitudes.

The comparison is:
1. NN-autodiff: learn only V(x) ~= u(t,x), then differentiate V.
2. ZOD one-sided: learn V, G, H, where G and H are fitted to one-sided ZOD
   derivative targets built from black-box terminal payoff queries at
   perturbed initial states x + eps Z.

Run:
    python scripts/spiky_value/run_spiky_value.py
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.quasirandom import SobolEngine


@dataclass(frozen=True)
class Metrics:
    value_rmse: float | None
    value_rrmse: float | None
    grad_rmse: float
    grad_rrmse: float
    hess_rmse: float
    hess_rrmse: float
    seconds: float


class SpikySDEValue:
    """Arithmetic Brownian value function with a spiky terminal payoff."""

    def __init__(
        self,
        sigma_sde: float,
        tau: float,
        variant: str,
        rough_scale: float,
        solution_scale: float,
        base_sin_freq: float,
        base_cos_freq: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.sigma_sde = sigma_sde
        self.tau = tau
        self.noise_std = sigma_sde * math.sqrt(tau)
        self.variant = variant
        self.solution_scale = solution_scale
        self.base_sin_freq = base_sin_freq
        self.base_cos_freq = base_cos_freq
        self.device = device
        self.dtype = dtype
        if variant == "needle_spiky":
            self.centers = torch.tensor(
                [-1.82, -1.48, -1.18, -0.91, -0.63, -0.34, -0.08, 0.16, 0.43, 0.69, 0.98, 1.29, 1.61],
                device=device,
                dtype=dtype,
            )
            self.widths = torch.tensor(
                [0.032, 0.026, 0.034, 0.028, 0.031, 0.025, 0.033, 0.027, 0.030, 0.026, 0.034, 0.029, 0.032],
                device=device,
                dtype=dtype,
            )
            self.amplitudes = torch.tensor(
                [0.020, -0.018, 0.019, -0.021, 0.017, 0.020, -0.018, 0.021, -0.019, 0.018, -0.020, 0.019, -0.017],
                device=device,
                dtype=dtype,
            )
        else:
            self.centers = torch.tensor(
                [-1.55, -1.05, -0.62, -0.18, 0.24, 0.71, 1.28],
                device=device,
                dtype=dtype,
            )
            self.widths = torch.tensor(
                [0.065, 0.055, 0.070, 0.060, 0.055, 0.065, 0.060],
                device=device,
                dtype=dtype,
            )
            self.amplitudes = torch.tensor(
                [0.035, -0.030, 0.032, 0.028, -0.034, 0.030, -0.027],
                device=device,
                dtype=dtype,
            )
        self.rough_freqs = torch.tensor([13.0, 21.0, 34.0, 47.0], device=device, dtype=dtype)
        self.rough_amps = rough_scale * torch.tensor([0.017, -0.012, 0.0085, -0.0065], device=device, dtype=dtype)
        self.rough_phases = torch.tensor([0.3, -0.8, 1.4, 2.1], device=device, dtype=dtype)

    def payoff(self, y: torch.Tensor) -> torch.Tensor:
        z = (y.unsqueeze(-1) - self.centers) / self.widths
        bumps = (self.amplitudes * torch.exp(-0.5 * z * z)).sum(dim=-1)
        base = 0.22 * torch.sin(self.base_sin_freq * y) + 0.06 * torch.cos(self.base_cos_freq * y)
        out = base + bumps
        if self.variant == "rough_periodic":
            phase = y.unsqueeze(-1) * self.rough_freqs + self.rough_phases
            out = out + (self.rough_amps * torch.sin(phase)).sum(dim=-1)
        return self.solution_scale * out

    def value_grad_hess(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-form value and spatial derivatives for X_T = x + sigma sqrt(tau) xi."""
        v2 = self.noise_std * self.noise_std

        sin_freq = self.base_sin_freq
        cos_freq = self.base_cos_freq
        sin_factor = math.exp(-0.5 * (sin_freq**2) * v2)
        cos_factor = math.exp(-0.5 * (cos_freq**2) * v2)
        value = 0.22 * sin_factor * torch.sin(sin_freq * x) + 0.06 * cos_factor * torch.cos(cos_freq * x)
        grad = 0.22 * sin_freq * sin_factor * torch.cos(sin_freq * x) - 0.06 * cos_freq * cos_factor * torch.sin(cos_freq * x)
        hess = -0.22 * sin_freq**2 * sin_factor * torch.sin(sin_freq * x) - 0.06 * cos_freq**2 * cos_factor * torch.cos(cos_freq * x)

        s2 = self.widths.pow(2) + v2
        scale = self.widths / torch.sqrt(s2)
        diff = x.unsqueeze(-1) - self.centers
        bump_value = self.amplitudes * scale * torch.exp(-0.5 * diff.pow(2) / s2)
        value = value + bump_value.sum(dim=-1)
        grad = grad + (bump_value * (-diff / s2)).sum(dim=-1)
        hess = hess + (bump_value * (diff.pow(2) / s2.pow(2) - 1.0 / s2)).sum(dim=-1)
        if self.variant == "rough_periodic":
            freqs = self.rough_freqs
            amps = self.rough_amps
            phases = self.rough_phases
            damp = torch.exp(-0.5 * freqs.pow(2) * v2)
            arg = x.unsqueeze(-1) * freqs + phases
            rough_value = amps * damp * torch.sin(arg)
            value = value + rough_value.sum(dim=-1)
            grad = grad + (amps * damp * freqs * torch.cos(arg)).sum(dim=-1)
            hess = hess - (amps * damp * freqs.pow(2) * torch.sin(arg)).sum(dim=-1)
        return self.solution_scale * value, self.solution_scale * grad, self.solution_scale * hess

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_grad_hess(x)[0]

    def terminal_from_state(self, x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return x + self.noise_std * xi


class MLP(nn.Module):
    def __init__(self, hidden: int, layers: int, x_min: float, x_max: float):
        super().__init__()
        self.register_buffer("x_mid", torch.tensor([(x_min + x_max) / 2.0], dtype=torch.float32))
        self.register_buffer("x_scale", torch.tensor([(x_max - x_min) / 2.0], dtype=torch.float32))
        modules: list[nn.Module] = []
        in_dim = 1
        for _ in range(layers):
            layer = nn.Linear(in_dim, hidden)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            modules.append(layer)
            modules.append(nn.Tanh())
            in_dim = hidden
        out = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        modules.append(out)
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(1)
        x_mid = self.x_mid.to(device=x.device, dtype=x.dtype)
        x_scale = self.x_scale.to(device=x.device, dtype=x.dtype)
        return self.net((x - x_mid) / x_scale).squeeze(1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_uniform(n: int, x_min: float, x_max: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return x_min + (x_max - x_min) * torch.rand(n, device=device, dtype=dtype)


def sample_training_points(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if args.zod_x_sampling == "grid":
        return torch.linspace(args.x_min, args.x_max, args.zod_target_points, device=device, dtype=dtype)
    if args.zod_x_sampling == "hybrid":
        n_grid = args.zod_target_points // 2
        n_random = args.zod_target_points - n_grid
        x_grid = torch.linspace(args.x_min, args.x_max, n_grid, device=device, dtype=dtype)
        x_random = sample_uniform(n_random, args.x_min, args.x_max, device, dtype)
        return torch.cat([x_grid, x_random], dim=0)
    return sample_uniform(args.zod_target_points, args.x_min, args.x_max, device, dtype)


def normal_from_uniform(u: torch.Tensor) -> torch.Tensor:
    # In float32, 1 - 1e-10 rounds back to 1.0, which makes erfinv(1)=inf.
    # Keep the clamp inside the representable open interval before applying
    # the inverse-normal transform to Sobol' points.
    eps = 1e-6 if u.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-12
    u = u.clamp(eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def draw_normals(
    n: int,
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sampler: str,
    seed: int,
    dimension: int = 1,
) -> torch.Tensor:
    if sampler == "rqmc":
        engine = SobolEngine(dimension=dimension, scramble=True, seed=seed)
        u = engine.draw(n * batch).to(device=device, dtype=dtype)
        z = normal_from_uniform(u)
        return z.reshape(n, batch) if dimension == 1 else z.reshape(n, batch, dimension)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(n, batch, device=device, dtype=dtype, generator=generator)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_value_net(
    target: SpikySDEValue,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, float]:
    model = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)
    opt = optim.Adam(model.parameters(), lr=args.lr_value)
    sync(device)
    t0 = time.perf_counter()
    model.train()
    for step in range(args.value_steps):
        x = sample_uniform(args.batch_size, args.x_min, args.x_max, device, dtype)
        if args.nn_value_target == "exact":
            y = target.value(x)
        else:
            xi = draw_normals(
                1,
                args.batch_size,
                device=device,
                dtype=dtype,
                sampler=args.noise_sampler,
                seed=args.seed + 1000003 + step,
            ).squeeze(0)
            y = target.payoff(target.terminal_from_state(x, xi)).detach()
        loss = torch.mean((model(x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            print(f"value step {step + 1}/{args.value_steps}: loss={loss.item():.4e}")
    sync(device)
    return model, time.perf_counter() - t0


def one_sided_zod_estimators(
    target: SpikySDEValue,
    x: torch.Tensor,
    eps: float,
    n_avg: int,
    chunk: int,
    z_sampler: str,
    noise_sampler: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-sided value/derivative estimators from black-box payoff observations."""
    value_sum = torch.zeros_like(x)
    grad_sum = torch.zeros_like(x)
    hess_sum = torch.zeros_like(x)
    done = 0
    batch = x.numel()
    while done < n_avg:
        c = min(chunk, n_avg - done)
        z = draw_normals(
            c,
            batch,
            device=x.device,
            dtype=x.dtype,
            sampler=z_sampler,
            seed=seed + 7919 + done,
        )
        xi = draw_normals(
            c,
            batch,
            device=x.device,
            dtype=x.dtype,
            sampler=noise_sampler,
            seed=seed + 104729 + done,
        )
        value_terminal = target.terminal_from_state(x.unsqueeze(0), xi)
        value_sum = value_sum + target.payoff(value_terminal).sum(dim=0)
        terminal = target.terminal_from_state(x.unsqueeze(0) + eps * z, xi)
        payoff = target.payoff(terminal)
        grad_sum = grad_sum + (z * payoff).sum(dim=0) / eps
        hess_sum = hess_sum + ((z * z - 1.0) * payoff).sum(dim=0) / (eps * eps)
        done += c
    return value_sum / n_avg, grad_sum / n_avg, hess_sum / n_avg


def one_sided_zod_targets(
    target: SpikySDEValue,
    x: torch.Tensor,
    eps: float,
    n_avg: int,
    chunk: int,
    z_sampler: str,
    noise_sampler: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided derivative targets used by the optional learned ZOD networks."""
    _, grad, hess = one_sided_zod_estimators(
        target,
        x,
        eps,
        n_avg,
        chunk,
        z_sampler,
        noise_sampler,
        seed,
    )
    return grad, hess


def train_zod_learner(
    target: SpikySDEValue,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module, nn.Module, float]:
    value_net = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)
    grad_net = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)
    hess_net = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)
    opt = optim.Adam(list(value_net.parameters()) + list(grad_net.parameters()) + list(hess_net.parameters()), lr=args.lr_zod)

    sync(device)
    t0 = time.perf_counter()
    print(
        "Building one-sided ZOD training dataset "
        f"(points={args.zod_target_points}, avg={args.zod_target_avg}, eps={args.eps}, "
        f"z={args.z_sampler}, xi={args.noise_sampler})"
    )
    x_data = sample_training_points(args, device, dtype)
    v_data = target.value(x_data)
    if args.zod_target_source == "analytic_os":
        smooth_sigma = math.sqrt(target.sigma_sde * target.sigma_sde + args.eps * args.eps)
        smooth_target = SpikySDEValue(
            smooth_sigma,
            target.tau,
            target.variant,
            args.rough_scale,
            target.solution_scale,
            target.base_sin_freq,
            target.base_cos_freq,
            device=device,
            dtype=dtype,
        )
        _, g_data, h_data = smooth_target.value_grad_hess(x_data)
    else:
        g_data, h_data = one_sided_zod_targets(
            target,
            x_data,
            args.eps,
            args.zod_target_avg,
            args.zod_chunk,
            args.z_sampler,
            args.noise_sampler,
            args.seed,
        )
    g_mean, g_std = g_data.mean(), g_data.std().clamp_min(1e-8)
    h_mean, h_std = h_data.mean(), h_data.std().clamp_min(1e-8)
    v_mean, v_std = v_data.mean(), v_data.std().clamp_min(1e-8)

    value_net.train()
    grad_net.train()
    hess_net.train()
    for step in range(args.zod_steps):
        idx = torch.randint(0, args.zod_target_points, (args.batch_size,), device=device)
        x = x_data[idx]
        v_tgt = v_data[idx]
        g_tgt = g_data[idx]
        h_tgt = h_data[idx]
        loss_v = torch.mean(((value_net(x) - v_mean) / v_std - (v_tgt - v_mean) / v_std) ** 2)
        loss_g = torch.mean(((grad_net(x) - g_mean) / g_std - (g_tgt - g_mean) / g_std) ** 2)
        loss_h = torch.mean(((hess_net(x) - h_mean) / h_std - (h_tgt - h_mean) / h_std) ** 2)
        loss = args.value_loss_weight * loss_v + args.grad_loss_weight * loss_g + args.hess_loss_weight * loss_h
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            print(
                f"zod step {step + 1}/{args.zod_steps}: "
                f"loss_v={loss_v.item():.3e}, loss_g={loss_g.item():.3e}, loss_h={loss_h.item():.3e}"
            )
    sync(device)
    return value_net, grad_net, hess_net, time.perf_counter() - t0


def train_zod_online_least_squares(
    target: SpikySDEValue,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module, nn.Module, float]:
    """Train V/G/H nets by online least-squares ZOD losses.

    This follows the style of ``zeroth.solver``: each optimization step samples
    fresh points and fresh payoff queries, rather than fitting a fixed
    precomputed derivative-target dataset.
    """
    value_net = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)
    grad_net = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)
    hess_net = MLP(args.hidden, args.layers, args.x_min, args.x_max).to(device=device, dtype=dtype)

    lr = args.lr_online if args.lr_online > 0 else args.lr_zod
    opt_v = optim.Adam(value_net.parameters(), lr=lr)
    opt_g = optim.Adam(grad_net.parameters(), lr=lr)
    opt_h = optim.Adam(hess_net.parameters(), lr=lr)

    batch_size = args.online_batch_size if args.online_batch_size > 0 else args.batch_size
    sync(device)
    t0 = time.perf_counter()

    for step in range(args.online_steps):
        x_v = sample_uniform(batch_size, args.x_min, args.x_max, device, dtype)
        if args.online_value_target == "exact":
            y_v = target.value(x_v).detach()
        else:
            xi_v = draw_normals(
                args.online_avg,
                batch_size,
                device=device,
                dtype=dtype,
                sampler=args.noise_sampler,
                seed=args.seed + 200003 + step,
            )
            y_v = target.payoff(target.terminal_from_state(x_v.unsqueeze(0), xi_v)).mean(dim=0).detach()

        loss_v = ((value_net(x_v) - y_v) ** 2).mean()
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()

        x = sample_uniform(batch_size, args.x_min, args.x_max, device, dtype)
        z = draw_normals(
            args.online_avg,
            batch_size,
            device=device,
            dtype=dtype,
            sampler=args.z_sampler,
            seed=args.seed + 300007 + step,
        )
        xi = draw_normals(
            args.online_avg,
            batch_size,
            device=device,
            dtype=dtype,
            sampler=args.noise_sampler,
            seed=args.seed + 400009 + step,
        )
        x_row = x.unsqueeze(0)

        if args.online_estimator == "mp":
            y_plus = target.payoff(target.terminal_from_state(x_row + args.eps * z, xi)).detach()
            y_minus = target.payoff(target.terminal_from_state(x_row - args.eps * z, xi)).detach()
            y_zero = target.payoff(target.terminal_from_state(x_row, xi)).detach()
            grad_tgt = (z * (y_plus - y_minus) / (2.0 * args.eps)).mean(dim=0)
            hess_tgt = (
                (z * z - 1.0)
                * (y_plus + y_minus - 2.0 * y_zero)
                / (2.0 * args.eps * args.eps)
            ).mean(dim=0)
        else:
            y_plus = target.payoff(target.terminal_from_state(x_row + args.eps * z, xi)).detach()
            grad_tgt = (z * y_plus / args.eps).mean(dim=0)
            hess_tgt = ((z * z - 1.0) * y_plus / (args.eps * args.eps)).mean(dim=0)

        loss_g = ((grad_net(x) - grad_tgt) ** 2).mean()
        loss_h = ((hess_net(x) - hess_tgt) ** 2).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        opt_h.zero_grad()
        loss_h.backward()
        opt_h.step()

        if args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            print(
                f"online LS step {step + 1}/{args.online_steps}: "
                f"V={loss_v.item():.3e}, G={loss_g.item():.3e}, H={loss_h.item():.3e}"
            )

    sync(device)
    return value_net, grad_net, hess_net, time.perf_counter() - t0


def nn_autodiff_derivatives(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    value = model(x_req)
    grad = torch.autograd.grad(value, x_req, torch.ones_like(value), create_graph=True, retain_graph=True)[0]
    hess = torch.autograd.grad(grad, x_req, torch.ones_like(grad), create_graph=False, retain_graph=False)[0]
    return value.detach(), grad.detach(), hess.detach()


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - true).pow(2))).item()


def rrmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    denom = torch.sqrt(torch.mean(true.pow(2))).item()
    return rmse(pred, true) / max(denom, 1e-12)


def make_metrics(
    value_pred: torch.Tensor | None,
    grad_pred: torch.Tensor,
    hess_pred: torch.Tensor,
    value_true: torch.Tensor,
    grad_true: torch.Tensor,
    hess_true: torch.Tensor,
    seconds: float,
) -> Metrics:
    value_rmse = None if value_pred is None else rmse(value_pred, value_true)
    value_rrmse = None if value_pred is None else rrmse(value_pred, value_true)
    return Metrics(
        value_rmse=value_rmse,
        value_rrmse=value_rrmse,
        grad_rmse=rmse(grad_pred, grad_true),
        grad_rrmse=rrmse(grad_pred, grad_true),
        hess_rmse=rmse(hess_pred, hess_true),
        hess_rrmse=rrmse(hess_pred, hess_true),
        seconds=seconds,
    )


def evaluate_nn_autodiff(model: nn.Module, target: SpikySDEValue, x: torch.Tensor, seconds: float) -> Metrics:
    true_v, true_g, true_h = target.value_grad_hess(x)
    pred_v, pred_g, pred_h = nn_autodiff_derivatives(model, x)
    return make_metrics(pred_v, pred_g, pred_h, true_v, true_g, true_h, seconds)


def evaluate_zod_learner(
    value_net: nn.Module,
    grad_net: nn.Module,
    hess_net: nn.Module,
    target: SpikySDEValue,
    x: torch.Tensor,
    seconds: float,
) -> Metrics:
    true_v, true_g, true_h = target.value_grad_hess(x)
    return make_metrics(value_net(x).detach(), grad_net(x).detach(), hess_net(x).detach(), true_v, true_g, true_h, seconds)


def evaluate_zod_formula(
    target: SpikySDEValue,
    x: torch.Tensor,
    args: argparse.Namespace,
    z_sampler: str | None = None,
) -> tuple[Metrics, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    sync(x.device)
    t0 = time.perf_counter()
    true_v, true_g, true_h = target.value_grad_hess(x)
    sampler = args.z_sampler if z_sampler is None else z_sampler
    pred_v, pred_g, pred_h = one_sided_zod_estimators(
        target,
        x,
        args.eps,
        args.eval_zod_avg,
        args.zod_chunk,
        sampler,
        args.noise_sampler,
        args.seed + 999,
    )
    sync(x.device)
    metrics = make_metrics(pred_v, pred_g, pred_h, true_v, true_g, true_h, time.perf_counter() - t0)
    return metrics, (pred_v.detach(), pred_g.detach(), pred_h.detach())


def sci_tex(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    base, exp = f"{value:.3e}".split("e")
    return rf"${float(base):.3f}\times10^{{{int(exp)}}}$"


def write_metrics(path: Path, rows: list[tuple[str, Metrics]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "value_rmse",
                "value_rrmse",
                "grad_rmse",
                "grad_rrmse",
                "hess_rmse",
                "hess_rrmse",
                "seconds",
            ]
        )
        for method, m in rows:
            writer.writerow(
                [
                    method,
                    "" if m.value_rmse is None else f"{m.value_rmse:.8e}",
                    "" if m.value_rrmse is None else f"{m.value_rrmse:.8e}",
                    f"{m.grad_rmse:.8e}",
                    f"{m.grad_rrmse:.8e}",
                    f"{m.hess_rmse:.8e}",
                    f"{m.hess_rrmse:.8e}",
                    f"{m.seconds:.4f}",
                ]
            )


def write_latex_table(path: Path, rows: list[tuple[str, Metrics]]) -> None:
    def fmt_optional(value: float | None) -> str:
        return "--" if value is None else sci_tex(value)

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Value rRMSE & Gradient rRMSE & Hessian rRMSE & Time (s) \\\\\n")
        f.write("\\midrule\n")
        for method, m in rows:
            f.write(
                f"{method} & {fmt_optional(m.value_rrmse)} & {sci_tex(m.grad_rrmse)} "
                f"& {sci_tex(m.hess_rrmse)} & ${m.seconds:.1f}$ \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def column_name(label: str) -> str:
    return (
        label.lower()
        .replace("-", "_")
        .replace(",", "")
        .replace(" ", "_")
        .replace("__", "_")
    )


def write_pointwise_results(
    path: Path,
    x: torch.Tensor,
    target: SpikySDEValue,
    nn_value_net: nn.Module,
    zod_curves: list[tuple[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> None:
    x_cpu = x.detach().cpu()
    true_v, true_g, true_h = [y.detach().cpu() for y in target.value_grad_hess(x)]
    nn_v, nn_g, nn_h = [y.detach().cpu() for y in nn_autodiff_derivatives(nn_value_net, x)]
    columns: list[tuple[str, torch.Tensor]] = [
        ("x", x_cpu),
        ("true_value", true_v),
        ("true_gradient", true_g),
        ("true_hessian", true_h),
        ("nn_value", nn_v),
        ("nn_value_error", nn_v - true_v),
        ("nn_value_abs_error", (nn_v - true_v).abs()),
        ("nn_gradient", nn_g),
        ("nn_gradient_error", nn_g - true_g),
        ("nn_gradient_abs_error", (nn_g - true_g).abs()),
        ("nn_hessian", nn_h),
        ("nn_hessian_error", nn_h - true_h),
        ("nn_hessian_abs_error", (nn_h - true_h).abs()),
    ]
    for label, values in zod_curves:
        prefix = column_name(label)
        zod_v, zod_g, zod_h = [y.detach().cpu() for y in values]
        columns.extend(
            [
                (f"{prefix}_value", zod_v),
                (f"{prefix}_value_error", zod_v - true_v),
                (f"{prefix}_value_abs_error", (zod_v - true_v).abs()),
                (f"{prefix}_gradient", zod_g),
                (f"{prefix}_gradient_error", zod_g - true_g),
                (f"{prefix}_gradient_abs_error", (zod_g - true_g).abs()),
                (f"{prefix}_hessian", zod_h),
                (f"{prefix}_hessian_error", zod_h - true_h),
                (f"{prefix}_hessian_abs_error", (zod_h - true_h).abs()),
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _ in columns])
        for i in range(x_cpu.numel()):
            writer.writerow([f"{values[i].item():.10e}" for _, values in columns])


def plot_results(
    path: Path,
    x: torch.Tensor,
    target: SpikySDEValue,
    nn_value_net: nn.Module,
    zod_curves: list[tuple[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> None:
    x_cpu = x.detach().cpu()
    true_v, true_g, true_h = [y.detach().cpu() for y in target.value_grad_hess(x)]
    nn_v, nn_g, nn_h = [y.detach().cpu() for y in nn_autodiff_derivatives(nn_value_net, x)]

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.4), sharex=True)
    panels = [
        (axes[0], true_v, nn_v, "Value", 0),
        (axes[1], true_g, nn_g, "Gradient", 1),
        (axes[2], true_h, nn_h, "Hessian", 2),
    ]
    colors = ["#0072B2", "#009E73", "#CC79A7"]
    linestyles = ["-", "-", "-."]
    for ax, true_y, nn_y, title, idx in panels:
        ax.plot(x_cpu, true_y, color="black", linewidth=1.2, label="exact")
        ax.plot(x_cpu, nn_y, color="#D55E00", linewidth=1.0, linestyle="--", label="NN-autodiff")
        for curve_idx, (label, values) in enumerate(zod_curves):
            zod_y = values[idx].detach().cpu()
            ax.plot(
                x_cpu,
                zod_y,
                color=colors[curve_idx % len(colors)],
                linewidth=1.0,
                linestyle=linestyles[curve_idx % len(linestyles)],
                label=label,
            )
        ax.set_title(title)
        ax.grid(True, color="#D0D0D0", linewidth=0.45, alpha=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=min(4, len(labels)),
    )
    axes[-1].set_xlabel("x")
    fig.subplots_adjust(top=0.90, hspace=0.34)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def learner_curve(
    value_net: nn.Module,
    grad_net: nn.Module,
    hess_net: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return value_net(x).detach(), grad_net(x).detach(), hess_net(x).detach()


def flatten_yaml_config(node: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in node.items():
        key = str(key)
        if isinstance(value, dict):
            flat.update(flatten_yaml_config(value, prefix=f"{prefix}{key}."))
        else:
            flat[key] = value
    return flat


def load_yaml_defaults(config_path: Path, parser: argparse.ArgumentParser) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")

    defaults = flatten_yaml_config(raw)
    valid_dests = {action.dest for action in parser._actions}
    unknown = sorted(key for key in defaults if key not in valid_dests)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in {config_path}: {', '.join(unknown)}. "
            "Use argument destination names such as 'payoff_variant', 'batch_size', or 'online_steps'."
        )
    return defaults


def write_run_config(path: Path, args: argparse.Namespace) -> None:
    payload: dict[str, object] = {}
    for key, value in sorted(vars(args).items()):
        payload[key] = str(value) if isinstance(value, Path) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "scripts":
        program_root = Path(__file__).resolve().parents[2]
        return program_root / path
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Optional YAML config. CLI arguments override YAML values.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x_min", type=float, default=-2.0)
    parser.add_argument("--x_max", type=float, default=2.0)
    parser.add_argument("--sigma_sde", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--payoff_variant", choices=["spiky", "rough_periodic", "needle_spiky"], default="spiky")
    parser.add_argument("--rough_scale", type=float, default=1.0)
    parser.add_argument("--solution_scale", type=float, default=1.0)
    parser.add_argument("--base_sin_freq", type=float, default=1.3)
    parser.add_argument("--base_cos_freq", type=float, default=4.7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--value_steps", type=int, default=9000)
    parser.add_argument("--zod_steps", type=int, default=9000)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--lr_zod", type=float, default=1e-3)
    parser.add_argument("--nn_value_target", choices=["exact", "payoff"], default="payoff")
    parser.add_argument("--eps", type=float, default=0.015)
    parser.add_argument("--zod_target_points", type=int, default=4096)
    parser.add_argument("--zod_target_avg", type=int, default=8192)
    parser.add_argument("--zod_chunk", type=int, default=512)
    parser.add_argument("--zod_x_sampling", choices=["random", "grid", "hybrid"], default="random")
    parser.add_argument("--zod_target_source", choices=["mc", "analytic_os"], default="mc")
    parser.add_argument("--value_loss_weight", type=float, default=1.0)
    parser.add_argument("--grad_loss_weight", type=float, default=1.0)
    parser.add_argument("--hess_loss_weight", type=float, default=1.0)
    parser.add_argument("--online_steps", type=int, default=12000)
    parser.add_argument("--online_batch_size", type=int, default=0)
    parser.add_argument("--online_avg", type=int, default=1)
    parser.add_argument("--lr_online", type=float, default=0.0)
    parser.add_argument("--online_estimator", choices=["op", "mp"], default="op")
    parser.add_argument("--online_value_target", choices=["payoff", "exact"], default="payoff")
    parser.add_argument("--eval_zod_avg", type=int, default=50000)
    parser.add_argument("--z_sampler", choices=["mc", "rqmc"], default="rqmc")
    parser.add_argument("--noise_sampler", choices=["mc", "rqmc"], default="mc")
    parser.add_argument("--test_points", type=int, default=2500)
    parser.add_argument("--print_freq", type=int, default=700)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--include_zod_learner", action="store_true")
    parser.add_argument("--zod_mode", choices=["direct", "learned", "online", "both"], default="direct")
    parser.add_argument("--compare_z_samplers", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=Path("scripts/spiky_value/results"))
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    config_args, remaining = pre_parser.parse_known_args()

    parser = build_parser()
    if config_args.config is not None:
        parser.set_defaults(**load_yaml_defaults(config_args.config, parser))
    args = parser.parse_args(remaining)
    args.config = config_args.config
    if not isinstance(args.out_dir, Path):
        args.out_dir = Path(args.out_dir)
    args.out_dir = resolve_output_dir(args.out_dir)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"device={device}, dtype={dtype}, seed={args.seed}")

    target = SpikySDEValue(
        args.sigma_sde,
        args.tau,
        args.payoff_variant,
        args.rough_scale,
        args.solution_scale,
        args.base_sin_freq,
        args.base_cos_freq,
        device=device,
        dtype=dtype,
    )
    nn_value_net, nn_seconds = train_value_net(target, args, device, dtype)

    x_test = torch.linspace(args.x_min, args.x_max, args.test_points, device=device, dtype=dtype)
    nn_metrics = evaluate_nn_autodiff(nn_value_net, target, x_test, nn_seconds)
    zod_curves: list[tuple[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []

    rows = [
        ("NN-autodiff", nn_metrics),
    ]
    if args.compare_z_samplers and args.zod_mode != "learned":
        for sampler in ("mc", "rqmc"):
            zod_formula_metrics, zod_formula_values = evaluate_zod_formula(target, x_test, args, z_sampler=sampler)
            rows.append((f"ZOD-OS, {sampler.upper()}", zod_formula_metrics))
            zod_curves.append((f"OS-{sampler.upper()}", zod_formula_values))
    elif args.zod_mode in ("direct", "both"):
        zod_formula_metrics, zod_formula_values = evaluate_zod_formula(target, x_test, args)
        rows.append((f"ZOD one-sided, {args.z_sampler.upper()}", zod_formula_metrics))
        zod_curves.append(("ZOD one-sided", zod_formula_values))
    if args.include_zod_learner or args.zod_mode in ("learned", "both"):
        set_seed(args.seed + 271828)
        zod_value_net, zod_grad_net, zod_hess_net, zod_seconds = train_zod_learner(target, args, device, dtype)
        zod_metrics = evaluate_zod_learner(zod_value_net, zod_grad_net, zod_hess_net, target, x_test, zod_seconds)
        rows.append(("ZOD one-sided", zod_metrics))
        zod_curves.append(("ZOD one-sided", learner_curve(zod_value_net, zod_grad_net, zod_hess_net, x_test)))
    if args.zod_mode == "online":
        set_seed(args.seed + 314159)
        zod_value_net, zod_grad_net, zod_hess_net, zod_seconds = train_zod_online_least_squares(target, args, device, dtype)
        zod_metrics = evaluate_zod_learner(zod_value_net, zod_grad_net, zod_hess_net, target, x_test, zod_seconds)
        zod_label = "ZOD one-point LS" if args.online_estimator == "op" else "ZOD multi-point LS"
        rows.append((zod_label, zod_metrics))
        zod_curves.append((zod_label, learner_curve(zod_value_net, zod_grad_net, zod_hess_net, x_test)))

    run_dir = args.out_dir / f"seed{args.seed}_eps{str(args.eps).replace('.', 'p')}"
    write_run_config(run_dir / "config.yaml", args)
    write_metrics(run_dir / "spiky_value_metrics.csv", rows)
    write_latex_table(run_dir / "spiky_value_table.tex", rows)
    write_pointwise_results(
        run_dir / "spiky_value_pointwise.csv",
        x_test,
        target,
        nn_value_net,
        zod_curves,
    )
    plot_results(
        run_dir / "spiky_value_comparison.pdf",
        x_test,
        target,
        nn_value_net,
        zod_curves,
    )

    print("\nMetrics")
    for method, m in rows:
        print(
            f"{method:24s} "
            f"value_rrmse={'' if m.value_rrmse is None else f'{m.value_rrmse:.3e}':>10s} "
            f"grad_rrmse={m.grad_rrmse:.3e} hess_rrmse={m.hess_rrmse:.3e} "
            f"time={m.seconds:.2f}s"
        )
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
