"""Equation definitions for the zeroth-order PDE solver.

Follows the same pattern as ``picard.equations``: an abstract base class that
exposes SDE sampling + PDE functions, with concrete implementations.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


class Equation(abc.ABC):
    r"""Base equation interface.

    The PDE has the general form:
        ∂_t u + (σ²/2) Δu + f(t, x, u, ∇u, ∇²u) = 0,  u(T, x) = g(x)

    The underlying SDE is  dX_t = σ dW_t  (arithmetic BM in ℝ^d).
    """

    # Set to False in subclasses whose f() does not use v_xx — skips expensive
    # Hessian computation during training target evaluation.
    has_hessian_term: bool = True
    has_gradient_term: bool = True

    def __init__(self, *, d: int = 1, sigma: float = 1.0, T: float = 1.0):
        self.d = d
        self.sigma = sigma
        self.T = T
        self._device: Optional[torch.device] = None

    # ---- device management (mirrors picard ParametersMixin) ----
    def to(self, *, device: torch.device):
        self._device = device
        return self

    @property
    def device(self):
        return self._device

    # ---- SDE sampling helpers ------------------------------------------
    def time_grid(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0.0, self.T, N + 1, device=device)

    @abc.abstractmethod
    def sample_x0(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample initial conditions X_0.  Returns (n, d)."""

    def sample_x(self, t: torch.Tensor) -> torch.Tensor:
        """Exact marginal: sample X_t | X_0=x0.  For arithmetic BM: X_t = x0 + σ√t · ε.
        t: (B, 1) on the correct device.  Returns (B, d).
        """
        B = t.size(0)
        x0 = self.sample_x0(B, t.device)  # (B, d)
        return x0 + self.sigma * torch.sqrt(t) * torch.randn(B, self.d, device=t.device)

    def sample_x_ts(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """One-step exact BM transition from (t, x) to time s.
        X_s = x + σ·√(s−t)·ε.  t, s: (B, 1);  x: (B, d) → (B, d).
        """
        dW = torch.randn_like(x)
        return x + self.sigma * torch.sqrt(s - t) * dW

    def sample_X_exact(
        self, N: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Full path sampling from 0 to T.  Returns (B, N+1, d)."""
        dt = self.T / N
        x0 = self.sample_x0(batch_size, device)  # (B, d)
        dW = torch.randn(batch_size, N, self.d, device=device) * np.sqrt(dt)
        increments = self.sigma * dW
        X = torch.cat(
            [x0.unsqueeze(1), x0.unsqueeze(1) + torch.cumsum(increments, dim=1)],
            dim=1,
        )
        return X  # (B, N+1, d)

    def sample_X_from_idx(
        self,
        idx_start: int,
        x_start: torch.Tensor,
        N: int,
        dw: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample path from t_{idx_start} to T starting at *x_start*."""
        if device is None:
            device = x_start.device
        x_start = x_start.to(device)
        B = x_start.size(0)
        N_remaining = N - idx_start

        if N_remaining <= 0:
            return x_start.unsqueeze(1) if x_start.dim() == 2 else x_start

        dt = self.T / N
        if dw is not None:
            dW = dw
        else:
            dW = torch.randn(B, N_remaining, self.d, device=device) * np.sqrt(dt)

        increments = self.sigma * dW
        x_start_2d = x_start if x_start.dim() == 2 else x_start.squeeze(1)
        X = torch.cat(
            [
                x_start_2d.unsqueeze(1),
                x_start_2d.unsqueeze(1) + torch.cumsum(increments, dim=1),
            ],
            dim=1,
        )
        return X  # (B, N_remaining+1, d)

    # ---- PDE interface -------------------------------------------------
    @abc.abstractmethod
    def g(self, x: torch.Tensor) -> torch.Tensor:
        """Terminal condition g(x).  x: (B, d) → (B, 1)."""

    @abc.abstractmethod
    def f(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        v_x: torch.Tensor,
        v_xx: torch.Tensor,
    ) -> torch.Tensor:
        """PDE source (nonlinear) term.  Returns (B, 1)."""

    @abc.abstractmethod
    def exact_solution(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """u*(t, x) when available.  (B, 1)."""

    @abc.abstractmethod
    def exact_derivatives(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (u, ∇u, ∇²u) — shapes (B,1), (B,d), (B,d,d)."""


# ---------------------------------------------------------------------------
# Concrete: LogisticSumBenchmark  (= "Cha" in picard, = LogisticSumBenchmarkPDE
# in the old zeroth_order code)
# ---------------------------------------------------------------------------
class LogisticSumBenchmark(Equation):
    r"""Logistic-sum benchmark PDE.

    PDE:
        ∂_t u + (σ²/2) Δu
        + [κσ²/√d · (u − 1/2) − 1/(κ√d)] · Σ_i ∂_{x_i} u = 0

    Terminal:  g(x) = sigmoid(T + κ/√d · Σ x_i)
    Exact:    u*(t,x) = sigmoid(t + κ/√d · Σ x_i)

    The nonlinear term f depends only on u and ∇u, not on ∇²u.
    """

    has_hessian_term = False

    def __init__(self, *, d: int = 20, kappa: float = 5.0, sigma: float = 1.0, T: float = 1.0):
        super().__init__(d=d, sigma=sigma, T=T)
        self.kappa = kappa
        self._alpha = kappa / np.sqrt(d)

    # -- SDE --
    def sample_x0(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, self.d, device=device)

    # -- terminal --
    def g(self, x: torch.Tensor) -> torch.Tensor:
        arg = self.T + self._alpha * x.sum(dim=1, keepdim=True)
        return torch.sigmoid(arg)

    # -- source term --
    def f(self, t, x, v, v_x, v_xx):
        grad_sum = v_x.sum(dim=1, keepdim=True)
        coeff = (
            self.kappa * (self.sigma ** 2) / np.sqrt(self.d) * (v - 0.5)
            - 1.0 / (self.kappa * np.sqrt(self.d))
        )
        return coeff * grad_sum

    # -- exact solution --
    def exact_solution(self, t, x):
        arg = t + self._alpha * x.sum(dim=1, keepdim=True)
        return torch.sigmoid(arg)

    def exact_derivatives(self, t, x):
        v = self.exact_solution(t, x)
        v_t = v * (1.0 - v)
        v_x = self._alpha * v_t * torch.ones_like(x)  # (B, d)

        B = x.size(0)
        beta = (self._alpha ** 2) * v_t * (1.0 - 2.0 * v)
        ones_mat = torch.ones(B, self.d, self.d, device=x.device, dtype=x.dtype)
        v_xx = beta.view(B, 1, 1) * ones_mat  # (B, d, d)
        return v, v_x, v_xx


class GBMEquationComplexExact(Equation):
    r"""Fully-nonlinear benchmark with an exact random-feature solution.

    PDE:
        \partial_t u + (alpha/2) \Delta u + F(t, x, \nabla^2 u) = 0

    where ``alpha = sigma^2`` and

        F(t, x, H) = 0.5*(1-alpha) * Tr(H)
                    + 0.25 * \sum_i |H_{ii}|
                    - u_t^*(t, x)
                    - 0.5 * \Delta u^*(t, x)
                    - 0.25 * \sum_i |u^*_{x_i x_i}(t, x)|.

    Exact solution is represented by random Fourier features:
        u*(t, x) = \sum_k v_k sin(w_{k,0} t + \sum_i w_{k,i} x_i).
    """

    has_hessian_term = True

    def __init__(
        self,
        *,
        d: int = 100,
        alpha: float = 1.0,
        T: float = 1.0,
        num_neurons: int = 2,
        w_path: Optional[str] = None,
        v_path: Optional[str] = None,
    ):
        sigma = float(np.sqrt(alpha))
        super().__init__(d=d, sigma=sigma, T=T)
        self.alpha = float(alpha)
        self.num_neurons = int(num_neurons)
        self.w_path = w_path
        self.v_path = v_path
        self.w, self.v = self._load_or_init_features()

    def to(self, *, device: torch.device):
        super().to(device=device)
        self.w = self.w.to(device=device)
        self.v = self.v.to(device=device)
        return self

    def _load_or_init_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w_file = Path(self.w_path) if self.w_path else None
        v_file = Path(self.v_path) if self.v_path else None

        if w_file is not None and v_file is not None and w_file.exists() and v_file.exists():
            w = torch.load(w_file, map_location="cpu")
            v = torch.load(v_file, map_location="cpu")
            return w, v

        w = torch.randn(self.num_neurons, 1 + self.d) * (1.0 / np.sqrt(self.d))
        w[:, 0] = 1.0
        v = torch.randn(self.num_neurons, 1)

        if w_file is not None and v_file is not None:
            w_file.parent.mkdir(parents=True, exist_ok=True)
            v_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(w, w_file)
            torch.save(v, v_file)

        return w, v

    def _tx(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_col = t * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return torch.cat([t_col, x], dim=1)

    def _wv_for(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.w.to(device=x.device, dtype=x.dtype)
        v = self.v.to(device=x.device, dtype=x.dtype)
        return w, v

    def sample_x0(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, self.d, device=device)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0], 1), self.T, device=x.device, dtype=x.dtype)
        return self.exact_solution(t, x)

    def exact_solution(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w, v = self._wv_for(x)
        tx = self._tx(t, x)
        return torch.sin(tx @ w.t()) @ v

    def u_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w, v = self._wv_for(x)
        tx = self._tx(t, x)
        return torch.cos(tx @ w.t()) @ (v * w[:, 0:1])

    def u_x(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w, v = self._wv_for(x)
        tx = self._tx(t, x)
        return torch.cos(tx @ w.t()) @ (v * w[:, 1:])

    def u_hessian(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w, v = self._wv_for(x)
        tx = self._tx(t, x)
        sin_term = -torch.sin(tx @ w.t())  # (B, m)
        outer = w[:, 1:].unsqueeze(2) * w[:, 1:].unsqueeze(1)  # (m, d, d)
        weights = v.unsqueeze(-1) * outer  # (m, d, d)
        return torch.einsum("bm,mij->bij", sin_term, weights)

    def laplacian(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w, v = self._wv_for(x)
        tx = self._tx(t, x)
        sin_term = torch.sin(tx @ w.t())
        return -sin_term @ (v * torch.sum(w[:, 1:] ** 2, dim=1, keepdim=True))

    def f(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        v_x: torch.Tensor,
        v_xx: torch.Tensor,
    ) -> torch.Tensor:
        u_ii = torch.diagonal(v_xx, dim1=1, dim2=2)
        laplacian = u_ii.sum(dim=1, keepdim=True)
        nonlinear = torch.abs(u_ii).sum(dim=1, keepdim=True)

        exact_u_t = self.u_t(t, x)
        exact_laplacian = self.laplacian(t, x)
        exact_diag_abs = torch.abs(torch.diagonal(self.u_hessian(t, x), dim1=1, dim2=2)).sum(
            dim=1, keepdim=True
        )

        return (
            0.5 * (1.0 - self.alpha) * laplacian
            + 0.25 * nonlinear
            - exact_u_t
            - 0.5 * exact_laplacian
            - 0.25 * exact_diag_abs
        )

    def exact_derivatives(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        val = self.exact_solution(t, x)
        grad = self.u_x(t, x)
        hess = self.u_hessian(t, x)
        return val, grad, hess


class OUProcessEquation(Equation):
    r"""HJB-style OU benchmark used in the paper experiments.

    PDE form used by the zeroth-order solver:

        ∂_t u + (alpha/2) Δu + f(t, x, u, ∇u) = 0,

    where

        f = -<theta*(mu - x), ∇u> - (alpha/2)||∇u||^2 - d*theta.

    Terminal condition is a negative log-likelihood of a Gaussian mixture.
    The exact solution at time t is -log p_{T-t}(x), where p_{tau} is the
    OU-evolved terminal mixture distribution.
    """

    has_hessian_term = False
    has_gradient_term = True

    def __init__(
        self,
        *,
        d: Optional[int] = None,
        nx: Optional[int] = None,
        alpha: float = 1.0,
        T: float = 1.0,
        theta: float = 1.0,
        mu: float = 0.0,
        num_components: int = 2,
        mean_scale: float = 1.0,
        var_scale: float = 2.0,
        alpha_scale: float = 4.0,
        mean_path: Optional[str] = None,
        var_path: Optional[str] = None,
        pi_path: Optional[str] = None,
    ):
        d_val = d if d is not None else nx
        if d_val is None:
            raise ValueError("Either d or nx must be provided for OUProcessEquation")

        alpha = float(alpha)
        super().__init__(d=int(d_val), sigma=float(np.sqrt(alpha)), T=float(T))

        self.alpha = alpha
        self.theta = float(theta)
        self.mu = float(mu)
        self.num_components = int(num_components)
        self.mean_scale = float(mean_scale)
        self.var_scale = float(var_scale)
        self.alpha_scale = float(alpha_scale)
        self.alpha_init = self.alpha_scale * self.alpha

        tag = f"{self.d}d_ms={self.mean_scale}_vs={self.var_scale}_{self.num_components}"
        self.mean_path = Path(mean_path) if mean_path is not None else Path(f"mean_{tag}.pt")
        self.var_path = Path(var_path) if var_path is not None else Path(f"var_{tag}.pt")
        self.pi_path = Path(pi_path) if pi_path is not None else Path(f"pi_{tag}.pt")

        self.mean0, self.var0, self.pi = self._load_or_init_components()

    @staticmethod
    def _to_cpu_tensor(x: torch.Tensor) -> torch.Tensor:
        return x.detach().clone().to(device="cpu")

    def _load_or_init_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mean_path.exists() and self.var_path.exists() and self.pi_path.exists():
            mean_raw = torch.load(self.mean_path, map_location="cpu")
            var_raw = torch.load(self.var_path, map_location="cpu")
            pi_raw = torch.load(self.pi_path, map_location="cpu")
        else:
            mean_raw = self.mean_scale * (torch.rand(self.num_components, self.d) * 2.0 - 1.0)
            eye = torch.eye(self.d)
            var_raw = self.var_scale * eye.unsqueeze(0).repeat(self.num_components, 1, 1)
            pi_raw = torch.rand(self.num_components)
            pi_raw = pi_raw / pi_raw.sum()

            self.mean_path.parent.mkdir(parents=True, exist_ok=True)
            self.var_path.parent.mkdir(parents=True, exist_ok=True)
            self.pi_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._to_cpu_tensor(mean_raw), self.mean_path)
            torch.save(self._to_cpu_tensor(var_raw), self.var_path)
            torch.save(self._to_cpu_tensor(pi_raw), self.pi_path)

        mean = mean_raw.float().reshape(self.num_components, self.d)
        pi = pi_raw.float().reshape(-1)
        pi = pi / pi.sum()

        if var_raw.ndim == 3:
            # Expect (K, d, d). Keep per-dimension diagonal variances.
            var_diag = torch.diagonal(var_raw.float(), dim1=1, dim2=2)
        elif var_raw.ndim == 2:
            # Accept (K, d) directly as diagonal variances.
            var_diag = var_raw.float()
        elif var_raw.ndim == 1:
            # Backward compatibility: isotropic variance per component.
            var_diag = var_raw.float().unsqueeze(1).expand(-1, self.d)
        else:
            raise ValueError(f"Unsupported var tensor shape: {tuple(var_raw.shape)}")

        var_diag = torch.clamp(var_diag.reshape(self.num_components, self.d), min=1e-8)
        if mean.shape[0] != self.num_components or var_diag.shape[0] != self.num_components or pi.shape[0] != self.num_components:
            raise ValueError("Loaded mean/var/pi shapes do not match num_components")

        return mean, var_diag, pi

    def to(self, *, device: torch.device):
        super().to(device=device)
        self.mean0 = self.mean0.to(device=device)
        self.var0 = self.var0.to(device=device)
        self.pi = self.pi.to(device=device)
        return self

    def sample_x0(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n, self.d, device=device) * float(np.sqrt(self.alpha_init))

    def _component_params(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tau: (B, 1)
        tau = tau.view(-1, 1)
        decay = torch.exp(-self.theta * tau)                # (B, 1)
        decay2 = decay * decay                              # (B, 1)

        mean = self.mu + (self.mean0.unsqueeze(0) - self.mu) * decay.unsqueeze(-1)  # (B, K, d)
        stationary_var = self.alpha / (2.0 * self.theta)
        var = self.var0.unsqueeze(0) * decay2.unsqueeze(-1) + stationary_var * (1.0 - decay2).unsqueeze(-1)  # (B, K, d)
        var = torch.clamp(var, min=1e-12)
        return mean, var

    def _gmm_log_prob(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # x: (B, d), mean: (B, K, d), var: (B, K, d)
        diff = x.unsqueeze(1) - mean                     # (B, K, d)
        log_w = torch.log(torch.clamp(self.pi, min=1e-12)).unsqueeze(0)  # (1, K)
        quad = ((diff * diff) / var).sum(dim=2)          # (B, K)
        log_det = torch.log(var).sum(dim=2)              # (B, K)
        log_norm = -0.5 * (
            self.d * np.log(2.0 * np.pi) + log_det + quad
        )
        return torch.logsumexp(log_w + log_norm, dim=1, keepdim=True)   # (B, 1)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        mean = self.mean0.unsqueeze(0).expand(B, -1, -1)
        var = self.var0.unsqueeze(0).expand(B, -1, -1)
        return -self._gmm_log_prob(x, mean, var)

    def f(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        v_x: torch.Tensor,
        v_xx: torch.Tensor,
    ) -> torch.Tensor:
        drift = self.theta * (self.mu - x)
        drift_term = -(drift * v_x).sum(dim=1, keepdim=True)
        quad_term = -0.5 * self.alpha * (v_x * v_x).sum(dim=1, keepdim=True)
        const_term = -self.d * self.theta * torch.ones_like(v)
        return drift_term + quad_term + const_term

    def exact_solution(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tau = self.T - t
        mean_tau, var_tau = self._component_params(tau)
        return -self._gmm_log_prob(x, mean_tau, var_tau)

    def exact_derivatives(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.size(0)
        x_req = x.detach().clone().requires_grad_(True)
        t_det = t.detach()

        with torch.enable_grad():
            v = self.exact_solution(t_det, x_req)
            grad = torch.autograd.grad(v.sum(), x_req, create_graph=True, retain_graph=True)[0]
            h_cols = []
            ones = torch.ones(B, device=x.device, dtype=x.dtype)
            for i in range(self.d):
                col = torch.autograd.grad(
                    grad[:, i], x_req,
                    grad_outputs=ones,
                    create_graph=False,
                    retain_graph=True,
                )[0]
                h_cols.append(col)
            hess = torch.stack(h_cols, dim=1)

        return v.detach(), grad.detach(), hess.detach()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
EQUATION_REGISTRY = {
    "LogisticSumBenchmark": LogisticSumBenchmark,
    "GBMEquationComplexExact": GBMEquationComplexExact,
    "OUProcessEquation": OUProcessEquation,
}


def build_equation(cls_name: str, **kwargs) -> Equation:
    if cls_name not in EQUATION_REGISTRY:
        raise ValueError(f"Unknown equation '{cls_name}'. Available: {list(EQUATION_REGISTRY.keys())}")
    return EQUATION_REGISTRY[cls_name](**kwargs)
