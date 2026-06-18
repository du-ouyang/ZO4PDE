"""Evaluation utilities — analogous to ``picard.evaluate``."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from zeroth.equations import Equation


class Evaluator:
    """Compute errors between neural-network predictions and the exact solution.

    Uses independent marginal sampling: for each time point t_j on a linspace
    grid, X_j ~ p_{t_j} is drawn independently via equation.sample_x(t_j).
    Memory cost is O(n_points * d) instead of O(N * n_points * d).
    """

    def __init__(
        self,
        equation: Equation,
        n_points: int = 10_000,
        eval_seed: int = 100041,
        device: Optional[torch.device] = None,
        hess_mode: str = "autograd",
        zod_eps: float = 0.02,
        zod_n_avg: int = 16,
    ):
        self.eq = equation
        self.n_points = n_points
        self.eval_seed = eval_seed
        self.device = device or torch.device("cpu")
        self.T = equation.T
        self.d = equation.d
        self.t_vec = torch.linspace(0, self.T, n_points, device=self.device)  # (n_points,)
        self.hess_mode = hess_mode
        self.zod_eps = zod_eps
        self.zod_n_avg = zod_n_avg

    # ---- seed context manager ----
    def _set_seed(self):
        torch.manual_seed(self.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.eval_seed)
        np.random.seed(self.eval_seed)

    @staticmethod
    def _hess_via_autograd(
        grad_net: nn.Module, inp: torch.Tensor, B: int, d: int
    ) -> torch.Tensor:
        """Compute Hessian of v by differentiating grad_net w.r.t. x (columns of inp[1:]).
        Returns (B, d, d).
        """
        x_part = inp[:, 1:].detach().requires_grad_(True)
        t_part = inp[:, :1].detach()
        inp2 = torch.cat([t_part, x_part], dim=1)
        g = grad_net(inp2)  # (B, d)
        hess_cols = []
        for i in range(d):
            col = torch.autograd.grad(
                g[:, i], x_part,
                grad_outputs=torch.ones(B, device=inp.device),
                create_graph=False, retain_graph=True,
            )[0]  # (B, d)
            hess_cols.append(col)
        return torch.stack(hess_cols, dim=1)  # (B, d, d)  row i = d/dx g_i

    def _hess_via_zod_fk(
        self,
        v_net: nn.Module,
        grad_net: nn.Module,
        hess_net: Optional[nn.Module],
        t: torch.Tensor,   # (B, 1)
        x: torch.Tensor,   # (B, d)
    ) -> torch.Tensor:     # (B, d, d)
        """Variance-reduced ZOD Hessian estimate via Feynman-Kac.

        Antithetic estimator averaged over self.zod_n_avg i.i.d. Z ~ N(0, I_d)::

            Ĥ = (1/n) Σ_z  (ZZᵀ − I) / (2ε²) · (Y⁺ + Y⁻ − 2Y₀)

        where Y± = FK(x ± ε·Z),  Y₀ = FK(x),  and BM noise (dW_T, dW_s) is
        shared across the three evaluations for variance reduction.

        FK(x_in) = g(X_T) + (T−t)·f(s, X_s, v(s,X_s), ∇v(s,X_s), H_v(s,X_s))
        with X_T = x_in + σ√(T−t)·dW_T,  X_s = x_in + σ√(s−t)·dW_s,
        and the nets evaluated at (s, X_s) to approximate the solution.

        Mixed-Precision Strategy
        ------------------------
        The numerical precision of Y⁺ + Y⁻ − 2Y₀ is the bottleneck for the ZOD
        Hessian estimate: each Yᵢ ≈ O(1), yet the signal is only ε²·Z⊤HZ ≈ O(ε²·‖H‖).
        In float32 (ε_mach ≈ 1.2e-7) the cancellation error is ≈ 2ε_mach/ε²:
        at ε=0.02 the relative error is ~0.6%; at ε=0.001 it exceeds 240%
        and the estimator is completely dominated by noise.

        Solution: network inference stays in float32 (limited by weight precision;
        RTX 3090 float64 throughput is only 1/64 of float32). _fk() casts its
        return value to float64 before returning, so the differencing Y⁺+Y⁻−2Y₀
        and all subsequent matrix operations run in float64. This reduces the
        cancellation error to ε_mach(f64)/ε² ≈ 5e-13 (at ε=0.02), which is
        negligible for any reasonable ε. The final result is cast back to float32
        to match the dtype of the rest of evaluate().
        """
        B = x.size(0)
        device = x.device
        f32 = torch.float32
        f64 = torch.float64
        eps = self.zod_eps

        # fixed seed so repeated evaluate() calls return identical estimates
        torch.manual_seed(self.eval_seed + 7777)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.eval_seed + 7777)

        # float32 tensors for network evaluation (weights are f32)
        t32 = t.to(f32)
        x32 = x.to(f32)
        T_all32    = torch.full((B, 1), self.T, device=device, dtype=f32)
        T_minus_t32 = T_all32 - t32  # (B, 1)

        def _fk(x_in32: torch.Tensor, dW_T32: torch.Tensor,
                s32: torch.Tensor, dW_s32: torch.Tensor) -> torch.Tensor:
            """Single-sample FK(x_in). Network inference in float32, returns (B,1) float64."""
            with torch.no_grad():
                X_T   = x_in32 + self.eq.sigma * torch.sqrt(T_minus_t32) * dW_T32
                g_val = self.eq.g(X_T)                               # (B, 1) f32
                X_s   = x_in32 + self.eq.sigma * torch.sqrt(s32 - t32) * dW_s32
                inp_s = torch.cat([s32, X_s], dim=1)
                v_s   = v_net(inp_s)                                 # (B, 1) f32
                gv_s  = grad_net(inp_s)                              # (B, d) f32
            if self.eq.has_hessian_term:
                if hess_net is not None:
                    with torch.no_grad():
                        hv_s = hess_net(inp_s).view(B, self.d, self.d)
                else:
                    hv_s = Evaluator._hess_via_autograd(grad_net, inp_s, B, self.d)
            else:
                hv_s = torch.zeros(B, self.d, self.d, device=device, dtype=f32)
            with torch.no_grad():
                f_val = self.eq.f(s32, X_s, v_s, gv_s, hv_s.detach())
                # cast to float64 before returning so the difference Y+Y--2Y0
                # is accumulated in double precision (avoids cancellation)
                return (g_val + T_minus_t32 * f_val).to(f64).detach()  # (B, 1) f64

        I_d     = torch.eye(self.d, device=device, dtype=f64).unsqueeze(0).expand(B, -1, -1)
        hess_sum = torch.zeros(B, self.d, self.d, device=device, dtype=f64)

        for _ in range(self.zod_n_avg):
            # perturbation direction and BM noise drawn in float32
            Z32   = torch.randn(B, self.d, device=device, dtype=f32)
            dW_T  = torch.randn(B, self.d, device=device, dtype=f32)
            s_smp = t32 + torch.rand(B, 1, device=device, dtype=f32) * T_minus_t32
            dW_s  = torch.randn(B, self.d, device=device, dtype=f32)

            Yp = _fk(x32 + eps * Z32, dW_T, s_smp, dW_s)   # (B, 1) f64
            Ym = _fk(x32 - eps * Z32, dW_T, s_smp, dW_s)   # (B, 1) f64
            Y0 = _fk(x32,             dW_T, s_smp, dW_s)   # (B, 1) f64

            # difference and outer-product in float64 — eliminates cancellation error
            Z64    = Z32.to(f64)
            ZZT    = Z64.unsqueeze(2) * Z64.unsqueeze(1)            # (B, d, d) f64
            Y_diff = (Yp + Ym - 2.0 * Y0).view(B, 1, 1)            # (B, 1, 1) f64
            hess_sum = hess_sum + (ZZT - I_d) * Y_diff / (2.0 * eps ** 2)

        # return in original dtype (float32) to match the rest of evaluate()
        return (hess_sum / self.zod_n_avg).to(x.dtype)  # (B, d, d)

    # ---- errors ----
    def evaluate(
        self,
        v_net: nn.Module,
        grad_net: nn.Module,
        hess_net: Optional[nn.Module],
    ) -> Dict[str, float]:
        """Evaluate errors.  If *hess_net* is None, Hessian is computed via
        self.hess_mode: ``'autograd'`` differentiates *grad_net* w.r.t. x;
        ``'zod_fk'`` uses the variance-reduced ZOD estimator on the
        Feynman-Kac target built from *v_net* and *grad_net*.

        Vectorised: sample all n_points (t_j, x_j) at once, single batched
        forward pass.  Each x_j is independently drawn from the marginal p_{t_j}.
        """
        v_net.eval(); grad_net.eval()
        if hess_net is not None:
            hess_net.eval()

        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        self._set_seed()

        # t: (n_points, 1),  x: (n_points, d)
        t_all = self.t_vec.unsqueeze(1)              # (n_points, 1)
        x_all = self.eq.sample_x(t_all)             # (n_points, d)
        inp = torch.cat([t_all, x_all], dim=1)      # (n_points, d+1)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state(gpu_state, self.device)
        np.random.set_state(np_state)

        B = self.n_points
        with torch.no_grad():
            pred_v = v_net(inp)      # (B, 1)
            pred_g = grad_net(inp)   # (B, d)

        if hess_net is not None:
            with torch.no_grad():
                pred_h = hess_net(inp).view(B, self.d, self.d)
        elif self.hess_mode == "zod_fk":
            pred_h = self._hess_via_zod_fk(v_net, grad_net, None, t_all, x_all)
        else:  # "autograd" (default)
            pred_h = self._hess_via_autograd(grad_net, inp, B, self.d)

        true_v, true_gx, true_hx = self.eq.exact_derivatives(t_all, x_all)

        # Riemann sum: mean over time points (uniform dt cancels in ratio, kept for absolute values)
        err_v = ((pred_v - true_v) ** 2).mean().item()
        err_g = ((pred_g - true_gx) ** 2).sum(dim=1).mean().item()
        err_h = ((pred_h - true_hx) ** 2).sum(dim=[1, 2]).mean().item()

        return {
            "value_error": err_v,
            "gradient_error": err_g,
            "hessian_error": err_h,
        }

    # ---- true-solution norms (for relative RMSE) ----
    def compute_true_norms(self) -> Dict[str, float]:
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        self._set_seed()

        t_all = self.t_vec.unsqueeze(1)          # (n_points, 1)
        x_all = self.eq.sample_x(t_all)         # (n_points, d)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state(gpu_state, self.device)
        np.random.set_state(np_state)

        true_v, true_gx, true_hx = self.eq.exact_derivatives(t_all, x_all)

        return {
            "value_norm": (true_v ** 2).mean().item(),
            "gradient_norm": (true_gx ** 2).sum(dim=1).mean().item(),
            "hessian_norm": (true_hx ** 2).sum(dim=[1, 2]).mean().item(),
        }
