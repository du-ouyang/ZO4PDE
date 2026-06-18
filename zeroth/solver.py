"""Zeroth-order Picard iteration runner."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from yacs.config import CfgNode

from zeroth.equations import Equation, build_equation
from zeroth.network import MLP, build_three_nets, build_value_net
from zeroth.evaluate import Evaluator
from zeroth.data_saver import DataSaver


class ZeroFunctionNet(nn.Module):
    """A non-trainable module representing the Picard initial zero function."""

    def __init__(self, output_dim: int, spatial_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.spatial_dim = int(spatial_dim)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        if self.output_dim == self.spatial_dim:
            # Tie the zero gradient to x so autograd can differentiate it.
            return tx[:, 1:] * 0.0
        zero = tx[:, :1] * 0.0
        return zero.expand(tx.shape[0], self.output_dim).clone()


class ZerothOrderRunner:
    """Orchestrates the full zeroth-order Picard iteration training loop.

    Runs the three-network Picard iteration used in the paper examples.
    """

    def __init__(self, cfg: CfgNode, config_path: Optional[str] = None):
        self.cfg = cfg
        self.config_path = Path(config_path).resolve() if config_path is not None else None
        self.output_root = self.config_path.parent if self.config_path is not None else None

        # ---- seed ----
        self.seed = cfg.SEED
        self._set_seed(self.seed)

        # ---- device / dtype ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ---- equation ----
        eq_kwargs = dict(cfg.EQUATION.kwargs) if cfg.EQUATION.kwargs else {}
        self.equation: Equation = build_equation(cfg.EQUATION.cls, **eq_kwargs)
        self.equation.to(device=self.device)
        self.d = self.equation.d

        # ---- solver hyper-params ----
        self.T = self.equation.T
        self.N = cfg.SOLVER.N
        self.num_iter = cfg.SOLVER.NUM_ITER
        self.num_train_steps = cfg.SOLVER.NUM_TRAIN_STEPS
        self.train_budget_switch_iter = int(cfg.SOLVER.TRAIN_BUDGET_SWITCH_ITER)
        self.num_train_steps_after_switch = int(cfg.SOLVER.NUM_TRAIN_STEPS_AFTER_SWITCH)
        self.lr_after_switch = float(cfg.SOLVER.LR_AFTER_SWITCH)
        self.eps = cfg.SOLVER.EPS
        self.lambda_deriv = cfg.SOLVER.LAMBDA_DERIV
        self.variance_reduction = cfg.SOLVER.VARIANCE_REDUCTION
        self.initial_function = str(cfg.SOLVER.INITIAL_FUNCTION).lower()
        if self.initial_function not in {"zero", "network"}:
            raise ValueError("SOLVER.INITIAL_FUNCTION must be either 'zero' or 'network'")
        self.pretrain_derivatives = bool(cfg.SOLVER.PRETRAIN_DERIVATIVES)
        self.pretrain_steps = cfg.SOLVER.PRETRAIN_STEPS
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.OPTIMIZER.lr

        # ---- networks ----
        self.input_dim = 1 + self.d
        hn = cfg.NETWORK.HIDDEN_SIZE
        nl = cfg.NETWORK.NUM_LAYERS
        act = cfg.NETWORK.ACTIVATION
        dr = cfg.NETWORK.DROPOUT
        sc = cfg.NETWORK.INIT_SCALE
        self.value_net_type = cfg.NETWORK.VALUE_NET.lower()
        self.train_hessian = cfg.NETWORK.TRAIN_HESSIAN
        self.has_hessian_term = self.equation.has_hessian_term
        if self.initial_function == "zero":
            self.V_net = ZeroFunctionNet(1, self.d).to(self.device)
            self.Grad_net = ZeroFunctionNet(self.d, self.d).to(self.device)
            self.Hess_net = (
                ZeroFunctionNet(self.d * self.d, self.d).to(self.device)
                if self.train_hessian
                else None
            )
        else:
            self.V_net, self.Grad_net, self.Hess_net = build_three_nets(
                self.input_dim, self.d, hn, nl, act, dr, sc, self.device,
                train_hessian=self.train_hessian,
                value_net_type=self.value_net_type,
                g0=self.equation.g,
                T=self.T,
            )

        # ---- evaluator ----
        self.evaluator = Evaluator(
            equation=self.equation,
            n_points=cfg.EVAL.N_POINTS,
            eval_seed=self.seed + 99999,
            device=self.device,
            hess_mode=cfg.EVAL.HESS_MODE,
            zod_eps=cfg.EVAL.ZOD_EPS,
            zod_n_avg=cfg.EVAL.ZOD_N_AVG,
        )
        self.eval_freq = max(1, int(cfg.EVAL.FREQ))

        # ---- error / metric history ----
        self.error_history: Dict[str, List[float]] = {
            "iteration": [],
            "value_error": [],
            "gradient_error": [],
            "hessian_error": [],
            "value_relative_rmse": [],
            "gradient_relative_rmse": [],
            "hessian_relative_rmse": [],
        }

        # network config cache for _new_network
        self._net_cfg = dict(
            hidden_size=hn, num_layers=nl, activation=act,
            dropout=dr, init_scale=1.0, value_net_type=self.value_net_type,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def _freeze(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze(net: nn.Module):
        for p in net.parameters():
            p.requires_grad = True

    @staticmethod
    def _copy_matching_parameters(dst: nn.Module, src: Optional[nn.Module]):
        """Warm-start dst from src when the two networks have compatible shapes."""
        if src is None:
            return
        with torch.no_grad():
            for p_new, p_old in zip(dst.parameters(), src.parameters()):
                if p_new.shape == p_old.shape:
                    p_new.copy_(p_old)

    def _new_network(self, output_dim: int, init_from: Optional[nn.Module] = None) -> Optional[nn.Module]:
        """Create a new network.  Returns None when output_dim == d*d and
        TRAIN_HESSIAN is False (two-net mode)."""
        if output_dim == self.d * self.d and not self.train_hessian:
            return None
        c = self._net_cfg
        hs = c["hidden_size"]
        if output_dim == 1:
            net = build_value_net(
                self.input_dim,
                hs,
                c["num_layers"],
                c["activation"],
                c["dropout"],
                c["init_scale"],
                self.device,
                value_net_type=c["value_net_type"],
                dim=self.d,
                g0=self.equation.g,
                T=self.T,
            )
            self._copy_matching_parameters(net, init_from)
            return net
        if output_dim == self.d:
            hs += 2 * self.d
        elif output_dim == self.d * self.d:
            hs += 2 * self.d * self.d
        net = MLP(self.input_dim, hs, c["num_layers"], output_dim,
                  c["activation"], c["dropout"], c["init_scale"]).to(self.device)
        self._copy_matching_parameters(net, init_from)
        return net

    def _training_budget_for_iteration(self, outer_iter: int) -> tuple[int, float]:
        """Return inner SGD steps and initial LR for a 1-based outer iteration."""
        steps = self.num_train_steps
        lr = self.lr
        if self.train_budget_switch_iter > 0 and outer_iter >= self.train_budget_switch_iter:
            if self.num_train_steps_after_switch > 0:
                steps = self.num_train_steps_after_switch
            if self.lr_after_switch > 0:
                lr = self.lr_after_switch
        return steps, lr

    # ------------------------------------------------------------------
    # pretrain derivative networks (auto-diff supervision)
    # ------------------------------------------------------------------
    def _pretrain_derivative_networks(
        self, V_net: nn.Module, Grad_net: nn.Module, Hess_net: nn.Module, num_steps: int
    ):
        print(f"\n{'='*60}\nPretraining derivative networks ({num_steps} steps)…\n{'='*60}")
        self._unfreeze(Grad_net)
        if Hess_net is not None:
            self._unfreeze(Hess_net)
        self._freeze(V_net)
        opt_G = optim.Adam(Grad_net.parameters(), lr=self.lr)
        opt_H = optim.Adam(Hess_net.parameters(), lr=self.lr) if Hess_net is not None else None
        sched_G = optim.lr_scheduler.StepLR(opt_G, step_size=1000, gamma=0.7)
        sched_H = optim.lr_scheduler.StepLR(opt_H, step_size=1000, gamma=0.7) if opt_H is not None else None

        bs = min(self.batch_size, 4096)

        for step in range(num_steps):
            # Sample (t, x) from exact marginal — no need for full discrete path
            t_s = torch.rand(bs, 1, device=self.device) * self.T
            x_s = self.equation.sample_x(t_s)   # (B, d)
            x_s.requires_grad_(True)
            tx = torch.cat([t_s, x_s], dim=1)
            v_pred = V_net(tx)

            # gradient
            grad_v = torch.autograd.grad(
                v_pred, x_s, torch.ones_like(v_pred),
                create_graph=False, retain_graph=True,
            )[0]
            loss_G = ((Grad_net(tx) - grad_v.detach()) ** 2).mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            # hessian (only in three-net mode)
            if Hess_net is not None:
                x_s2 = x_s.detach().clone().requires_grad_(True)
                tx2 = torch.cat([t_s, x_s2], dim=1)
                v2 = V_net(tx2)
                grad_v2 = torch.autograd.grad(v2, x_s2, torch.ones_like(v2), create_graph=True, retain_graph=True)[0]
                hess_cols = []
                for i in range(self.d):
                    col = torch.autograd.grad(
                        grad_v2[:, i], x_s2, torch.ones(bs, device=self.device),
                        create_graph=False, retain_graph=True,
                    )[0]
                    hess_cols.append(col)
                hess_true = torch.stack(hess_cols, dim=1).view(bs, -1)  # (B, d*d)
                loss_H = ((Hess_net(tx2) - hess_true.detach()) ** 2).mean()
                opt_H.zero_grad(); loss_H.backward(); opt_H.step()

            sched_G.step()
            if sched_H is not None:
                sched_H.step()
            if (step + 1) % 500 == 0:
                h_str = f" Hess={loss_H.item():.4e}" if Hess_net is not None else ""
                print(f"  Pretrain {step+1}/{num_steps} | Grad={loss_G.item():.4e}{h_str}")

        self._freeze(Grad_net)
        if Hess_net is not None:
            self._freeze(Hess_net)
        h_str = f"  Hess={loss_H.item():.4e}" if Hess_net is not None else ""
        print(f"Pretraining done. Grad={loss_G.item():.4e}{h_str}\n")

    # ------------------------------------------------------------------
    # single-sample integral estimator
    # ------------------------------------------------------------------
    def _eval_f_integral(
        self, s: torch.Tensor, X_s: torch.Tensor, T_minus_t: torch.Tensor
    ) -> torch.Tensor:
        """Unbiased estimate  (T−t) · f(s, X_s, ...)  using the frozen prev-iter nets.

        Args:
            s         : (B, 1) sampled time in [t, T]
            X_s       : (B, d) BM state at time s
            T_minus_t : (B, 1) = T − t
        Returns:
            (B, 1)
        """
        B = X_s.size(0)
        inp_s = torch.cat([s, X_s], dim=1)
        with torch.no_grad():
            v_s = self.V_net(inp_s)
            gv_s = self.Grad_net(inp_s)
            if self.has_hessian_term and self.Hess_net is not None:
                hv_s = self.Hess_net(inp_s).view(B, self.d, self.d)
            elif not self.has_hessian_term:
                hv_s = torch.zeros(B, self.d, self.d, device=X_s.device)

        if self.has_hessian_term and self.Hess_net is None:
            # two-net mode: Hessian is obtained by differentiating Grad_net
            # and therefore must run with grad enabled.
            with torch.enable_grad():
                hv_s = self._autograd_hessian(self.Grad_net, inp_s)

        with torch.no_grad():
            f_val = self.equation.f(s, X_s, v_s, gv_s, hv_s.detach())
        return T_minus_t * f_val  # (B, 1)

    @staticmethod
    def _autograd_hessian(grad_net: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        """Compute Hessian of v w.r.t. x via autograd of grad_net.  Returns (B, d, d)."""
        B, d_plus_1 = inp.shape
        d = d_plus_1 - 1
        x_part = inp[:, 1:].detach().requires_grad_(True)
        t_part = inp[:, :1].detach()
        g = grad_net(torch.cat([t_part, x_part], dim=1))  # (B, d)
        hess_cols = []
        for i in range(d):
            col = torch.autograd.grad(
                g[:, i], x_part,
                grad_outputs=torch.ones(B, device=inp.device),
                create_graph=False, retain_graph=True,
            )[0]
            hess_cols.append(col)
        return torch.stack(hess_cols, dim=1)  # (B, d, d)

    # ------------------------------------------------------------------
    # zeroth-order derivative loss
    # ------------------------------------------------------------------
    def _zeroth_order_derivative_loss(
        self,
        v_net: nn.Module,
        grad_net: nn.Module,
        hess_net: Optional[nn.Module],
        t: torch.Tensor,
        x: torch.Tensor,
    ):
        """Estimate \u2207u and \u2207\u00b2u via zeroth-order finite differences on the
        Feynman-Kac target, using exact BM marginal transitions (no discretisation).
        """
        B = x.size(0)
        device = x.device
        inp = torch.cat([t, x], dim=1)
        v_x_nn = grad_net(inp)  # (B, d)

        Z = torch.randn(B, self.d, device=device)

        # Pre-draw shared BM noise so antithetic x+/x− cancel properly
        T_tensor  = torch.full((B, 1), self.T, device=device)
        T_minus_t = T_tensor - t                         # (B, 1)
        dW_T = torch.randn(B, self.d, device=device)     # noise for terminal
        s    = t + torch.rand_like(t) * T_minus_t        # s ~ U[t, T]
        dW_s = torch.randn(B, self.d, device=device)     # noise for integral

        def feynman_kac(x_in: torch.Tensor) -> torch.Tensor:
            """Single-sample Feynman-Kac: g(X_T) + (T−t)·f(s, X_s).  No grad."""
            X_T = x_in + self.equation.sigma * torch.sqrt(T_minus_t) * dW_T
            g_val = self.equation.g(X_T)
            X_s = x_in + self.equation.sigma * torch.sqrt(s - t) * dW_s
            return g_val + self._eval_f_integral(s, X_s, T_minus_t)

        if not self.variance_reduction:
            Yp = feynman_kac(x + self.eps * Z)
            zo_grad = Z / self.eps * Yp
            if hess_net is not None:
                ZZT = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))
                I_d  = torch.eye(self.d, device=device).unsqueeze(0).expand(B, -1, -1)
                zo_hess = (-I_d + ZZT) / (self.eps ** 2) * Yp.unsqueeze(2)
        else:
            Yp = feynman_kac(x + self.eps * Z)
            Ym = feynman_kac(x - self.eps * Z)
            zo_grad = Z / (2 * self.eps) * (Yp - Ym)
            if hess_net is not None:
                Y0  = feynman_kac(x)
                ZZT = torch.bmm(Z.unsqueeze(2), Z.unsqueeze(1))
                I_d  = torch.eye(self.d, device=device).unsqueeze(0).expand(B, -1, -1)
                Y_diff = (Yp + Ym - 2 * Y0).unsqueeze(2)
                zo_hess = (-I_d + ZZT) / (2 * self.eps ** 2) * Y_diff

        loss_grad = ((zo_grad - v_x_nn) ** 2).mean()
        if hess_net is not None:
            v_xx_nn   = hess_net(inp).view(B, self.d, self.d)
            loss_hess = ((zo_hess - v_xx_nn) ** 2).mean()
        else:
            loss_hess = torch.tensor(0.0, device=device)
        return loss_grad, loss_hess

    # ------------------------------------------------------------------
    # main run
    # ------------------------------------------------------------------
    def run(self):
        print(f"\n{'='*60}")
        print(f"Zeroth-Order Solver  |  d={self.d}  iters={self.num_iter}")
        mode = "3-net" if self.train_hessian else "2-net (Hessian via autograd)"
        print(f"Equation: {self.cfg.EQUATION.cls}  mode: {mode}  device: {self.device}")
        print(f"Picard initial function: {self.initial_function}")
        if self.initial_function == "zero" and self.pretrain_derivatives:
            print("Derivative pretraining is skipped because INITIAL_FUNCTION=zero.")
        print(f"{'='*60}\n")

        # experiment directory
        saver = DataSaver(
            self.cfg,
            output_root=str(self.output_root) if self.output_root is not None else None,
        )
        self.exp_dir = saver.exp_dir

        # compute true-solution norms for relative RMSE
        true_norms = self.evaluator.compute_true_norms()

        t_total = time.time()

        if self.initial_function == "zero":
            print("Recording iteration 0 from the Picard zero initial function...")
            errs = self.evaluator.evaluate(self.V_net, self.Grad_net, self.Hess_net)
            self._record_errors(0, errs, true_norms)
            saver.save_error_history(self.error_history)

        for n in range(self.num_iter):
            outer_iter = n + 1
            num_train_steps_iter, lr_iter = self._training_budget_for_iteration(outer_iter)
            t_iter = time.time()
            print(f"\n{'='*50}  Iteration {outer_iter}/{self.num_iter}  {'='*50}")
            print(f"Training budget: steps={num_train_steps_iter}  initial_lr={lr_iter:.2e}")

            self._freeze(self.V_net)
            self._freeze(self.Grad_net)
            if self.Hess_net is not None:
                self._freeze(self.Hess_net)

            V_next = self._new_network(1, init_from=self.V_net)
            Grad_next = self._new_network(self.d, init_from=self.Grad_net)
            Hess_next = self._new_network(self.d * self.d, init_from=self.Hess_net)  # None in two-net mode

            # Legacy mode: pretrain derivative nets before the first Picard update.
            if (
                n == 0
                and self.initial_function == "network"
                and self.pretrain_derivatives
                and self.pretrain_steps > 0
            ):
                self._pretrain_derivative_networks(V_next, Grad_next, Hess_next, self.pretrain_steps)

                self.V_net = V_next
                self.Grad_net = Grad_next
                self.Hess_net = Hess_next

                errs = self.evaluator.evaluate(self.V_net, self.Grad_net, self.Hess_net)
                self._record_errors(0, errs, true_norms)

                self._freeze(self.V_net)
                self._freeze(self.Grad_net)
                if self.Hess_net is not None:
                    self._freeze(self.Hess_net)
                V_next = self._new_network(1, init_from=self.V_net)
                Grad_next = self._new_network(self.d, init_from=self.Grad_net)
                Hess_next = self._new_network(self.d * self.d, init_from=self.Hess_net)  # None in two-net mode

            self._unfreeze(V_next)
            self._unfreeze(Grad_next)
            if Hess_next is not None:
                self._unfreeze(Hess_next)

            opt_V = optim.Adam(V_next.parameters(), lr=lr_iter)
            opt_G = optim.Adam(Grad_next.parameters(), lr=lr_iter)
            opt_H = optim.Adam(Hess_next.parameters(), lr=lr_iter) if Hess_next is not None else None
            sc = self.cfg.TRAIN.SCHEDULER
            sch_V = optim.lr_scheduler.StepLR(opt_V, sc.V_step_size, sc.V_gamma)
            sch_G = optim.lr_scheduler.StepLR(opt_G, sc.Grad_step_size, sc.Grad_gamma)
            sch_H = optim.lr_scheduler.StepLR(opt_H, sc.Hess_step_size, sc.Hess_gamma) if opt_H is not None else None

            for it in range(num_train_steps_iter):
                # Exact marginal: t ~ U[0, T],  x ~ p(X_t | X_0=x0)
                t_s = torch.rand(self.batch_size, 1, device=self.device) * self.T
                x_s = self.equation.sample_x(t_s)  # (B, d), no discretisation

                # --- value loss: Feynman-Kac (unbiased single-sample) ---
                T_tensor  = torch.full((self.batch_size, 1), self.T, device=self.device)
                T_minus_t = T_tensor - t_s
                # terminal  g(X_T),  X_T ~ BM one-step from (t_s, x_s)
                X_T   = self.equation.sample_x_ts(t_s, T_tensor, x_s)
                g_val = self.equation.g(X_T)
                # integral  (T-t)·f(s, X_s),  s ~ U[t, T]
                s_int   = t_s + torch.rand_like(t_s) * T_minus_t
                X_s_int = self.equation.sample_x_ts(t_s, s_int, x_s)
                int_f   = self._eval_f_integral(s_int, X_s_int, T_minus_t)
                target  = (g_val + int_f).detach()
                pred    = V_next(torch.cat([t_s, x_s], dim=1))
                loss_v  = ((pred - target) ** 2).mean()
                opt_V.zero_grad(); loss_v.backward(); opt_V.step()

                # --- derivative loss ---
                loss_g, loss_h = self._zeroth_order_derivative_loss(V_next, Grad_next, Hess_next, t_s, x_s)
                opt_G.zero_grad(); loss_g.backward(); opt_G.step()
                if opt_H is not None and loss_h.requires_grad:
                    opt_H.zero_grad(); loss_h.backward(); opt_H.step()

                sch_V.step(); sch_G.step()
                if sch_H is not None:
                    sch_H.step()

                if (it + 1) % self.cfg.LOGGING.PRINT_FREQ == 0:
                    print(f"  step {it+1}/{num_train_steps_iter} | "
                          f"V={loss_v.item():.4e} G={loss_g.item():.4e} H={loss_h.item():.4e} "
                          f"lr={opt_V.param_groups[0]['lr']:.2e}")

            print(f"Final: V={loss_v.item():.4e}  G={loss_g.item():.4e}  H={loss_h.item():.4e}")

            self.V_net = V_next
            self.Grad_net = Grad_next
            self.Hess_net = Hess_next
            self._freeze(self.V_net)
            self._freeze(self.Grad_net)
            if self.Hess_net is not None:
                self._freeze(self.Hess_net)

            is_final_iter = (n == self.num_iter - 1)
            saver.save_checkpoint(self.V_net, self.Grad_net, self.Hess_net, name=f"model_iter_{n+1}")

            dt_iter = time.time() - t_iter
            should_eval = is_final_iter or ((n + 1) % self.eval_freq == 0)
            if should_eval:
                if is_final_iter:
                    print("Final iteration done. Evaluating final model...")
                else:
                    print(f"Iteration done. Evaluating current model (freq={self.eval_freq})...")

                errs = self.evaluator.evaluate(self.V_net, self.Grad_net, self.Hess_net)
                self._record_errors(n + 1, errs, true_norms)
                saver.save_error_history(self.error_history)

                h_str = f"  HessErr={errs['hessian_error']:.4e}" if errs.get('hessian_error') is not None else ""
                print(f"  ValErr={errs['value_error']:.4e}  GradErr={errs['gradient_error']:.4e}{h_str}  ({dt_iter:.1f}s)")
            else:
                print(f"Iteration done. Skipping evaluation this round (freq={self.eval_freq})  ({dt_iter:.1f}s)")

        total = time.time() - t_total
        print(f"\n{'='*50}\nDone! Total time {total:.1f}s ({total/60:.1f} min)\n{'='*50}")
        saver.save_checkpoint(self.V_net, self.Grad_net, self.Hess_net, name="final_model")
        saver.save_error_history(self.error_history)
        saver.save_report(self.cfg, self.error_history, total, self.num_iter)

    # ------------------------------------------------------------------
    def _record_errors(self, iteration, errs, norms):
        self.error_history["iteration"].append(iteration)
        for k in ("value_error", "gradient_error", "hessian_error"):
            self.error_history[k].append(errs[k])
        for k_err, k_norm, k_rel in [
            ("value_error", "value_norm", "value_relative_rmse"),
            ("gradient_error", "gradient_norm", "gradient_relative_rmse"),
            ("hessian_error", "hessian_norm", "hessian_relative_rmse"),
        ]:
            n = norms[k_norm]
            self.error_history[k_rel].append(
                np.sqrt(errs[k_err] / n) if n > 0 else float("inf")
            )
        rel = self.error_history
        print(f"  Rel RMSE: V={rel['value_relative_rmse'][-1]:.4e}  "
              f"G={rel['gradient_relative_rmse'][-1]:.4e}  "
              f"H={rel['hessian_relative_rmse'][-1]:.4e}")
