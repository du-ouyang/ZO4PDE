"""Persistence helpers: save checkpoints, error histories, and reports."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from yacs.config import CfgNode


class DataSaver:
    """Manages an experiment directory and provides save utilities.

    Mirrors the role of ``picard.data_saver`` plus the report functionality
    that was previously scattered across ``solver_multidim_three_nets.py``.
    """

    def __init__(self, cfg: CfgNode, output_root: Optional[str] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(output_root) if output_root is not None else Path("results")
        self.exp_dir = str(base_dir / f"{cfg.NAME}_{self.timestamp}")
        if cfg.FORCE and os.path.exists(self.exp_dir):
            import shutil
            shutil.rmtree(self.exp_dir)
        os.makedirs(self.exp_dir, exist_ok=True)

        # dump config
        cfg_path = os.path.join(self.exp_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(cfg.dump())
        print(f"Config saved -> {cfg_path}")

    # ---- checkpoint ----
    def save_checkpoint(
        self, v_net: nn.Module, grad_net: nn.Module, hess_net: nn.Module, name: str = "final_model"
    ):
        path = os.path.join(self.exp_dir, f"{name}.pt")
        torch.save(
            {
                "v_net": v_net.state_dict(),
                "grad_net": grad_net.state_dict(),
                "hess_net": hess_net.state_dict() if hess_net is not None else None,
            },
            path,
        )
        print(f"Checkpoint -> {path}")

    # ---- error history ----
    def save_error_history(self, history: Dict[str, List[float]]):
        df = pd.DataFrame(history)
        csv_path = os.path.join(self.exp_dir, "error_history.csv")
        df.to_csv(csv_path, index=False)

    # ---- training report ----
    def save_report(
        self,
        cfg: CfgNode,
        history: Dict[str, List[float]],
        total_time: float,
        num_iter: int,
    ):
        path = os.path.join(self.exp_dir, "training_report.txt")
        with open(path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING REPORT (Zeroth-Order Three-Net Solver)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment:  {cfg.NAME}\n")
            f.write(f"Timestamp:   {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Equation:    {cfg.EQUATION.cls}\n")
            f.write(f"Dimension:   d={cfg.EQUATION.kwargs.get('d', '?')}\n\n")

            f.write("Configuration\n" + "-" * 60 + "\n")
            f.write(cfg.dump())
            f.write("\n")

            f.write("Timing\n" + "-" * 60 + "\n")
            f.write(f"  Total: {total_time:.1f}s  ({total_time/60:.1f} min)\n")
            f.write(f"  Avg/iter: {total_time/max(num_iter,1):.1f}s\n\n")

            f.write("Final errors\n" + "-" * 60 + "\n")
            for k in ("value_error", "gradient_error", "hessian_error"):
                f.write(f"  {k}: {history[k][-1]:.6e}\n")
            f.write("\n")
            for k in ("value_relative_rmse", "gradient_relative_rmse", "hessian_relative_rmse"):
                f.write(f"  {k}: {history[k][-1]:.6e}\n")
        print(f"Report -> {path}")
