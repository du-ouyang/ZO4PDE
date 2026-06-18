"""YACS-based hierarchical configuration for the zeroth-order PDE solver."""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
from yacs.config import CfgNode

_C = CfgNode()

_C.BASE = None
_C.FORCE = False  # whether to force overwrite the existing directory
_C.NAME = "exp"

# ---- equation ----
_C.EQUATION = CfgNode()
_C.EQUATION.cls = "LogisticSumBenchmark"
_C.EQUATION.kwargs = CfgNode(new_allowed=True)

# ---- zeroth-order solver ----
_C.SOLVER = CfgNode()
_C.SOLVER.N = 80         # number of time steps for SDE path
_C.SOLVER.NUM_ITER = 10  # number of Picard (outer) iterations
_C.SOLVER.NUM_TRAIN_STEPS = 20000  # SGD steps per iteration
_C.SOLVER.TRAIN_BUDGET_SWITCH_ITER = 0  # 0: disabled; k: use below from outer iter k onward
_C.SOLVER.NUM_TRAIN_STEPS_AFTER_SWITCH = 0  # <=0: keep NUM_TRAIN_STEPS
_C.SOLVER.LR_AFTER_SWITCH = 0.0  # <=0: keep TRAIN.OPTIMIZER.lr
_C.SOLVER.EPS = 0.001    # perturbation size for zeroth-order derivative
_C.SOLVER.LAMBDA_DERIV = 1.0
_C.SOLVER.VARIANCE_REDUCTION = True
_C.SOLVER.INITIAL_FUNCTION = "network"  # "network": legacy random-net initial state; "zero": Picard u^0=0
_C.SOLVER.PRETRAIN_DERIVATIVES = True  # legacy derivative pretrain before the first Picard step
_C.SOLVER.PRETRAIN_STEPS = 5000  # used only when PRETRAIN_DERIVATIVES is true

# ---- training ----
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 4096
_C.TRAIN.OPTIMIZER = CfgNode()
_C.TRAIN.OPTIMIZER.cls = "Adam"
_C.TRAIN.OPTIMIZER.lr = 5e-3
_C.TRAIN.SCHEDULER = CfgNode()
_C.TRAIN.SCHEDULER.V_step_size = 1000
_C.TRAIN.SCHEDULER.V_gamma = 0.5
_C.TRAIN.SCHEDULER.Grad_step_size = 2000
_C.TRAIN.SCHEDULER.Grad_gamma = 0.5
_C.TRAIN.SCHEDULER.Hess_step_size = 2000
_C.TRAIN.SCHEDULER.Hess_gamma = 0.5

# ---- network ----
_C.NETWORK = CfgNode()
_C.NETWORK.HIDDEN_SIZE = 128
_C.NETWORK.NUM_LAYERS = 4
_C.NETWORK.ACTIVATION = "tanh"
_C.NETWORK.DROPOUT = 0.0
_C.NETWORK.INIT_SCALE = 1.0
_C.NETWORK.VALUE_NET = "mlp"  # "mlp" or "pisgradnet"
_C.NETWORK.TRAIN_HESSIAN = True  # False: skip Hess_net; eval hessian via autograd of Grad_net

# ---- evaluation ----
_C.EVAL = CfgNode()
_C.EVAL.N_POINTS = 10_000  # number of time points for independent marginal sampling
_C.EVAL.FREQ = 1           # evaluate every `FREQ` iterations
_C.EVAL.HESS_MODE = "autograd"  # "autograd": diff Grad_net | "zod_fk": ZOD on Feynman-Kac target
_C.EVAL.ZOD_N_AVG = 16          # number of Z draws to average in zod_fk mode
_C.EVAL.ZOD_EPS   = 0.02        # perturbation size for zod_fk (optimal for float32: ~eps_mach^{1/4})

# ---- logging ----
_C.LOGGING = CfgNode()
_C.LOGGING.PRINT_FREQ = 1000  # print every N training steps

# ---- reproducibility ----
_C.SEED = 42

# ---- data ----
_C.DATA = CfgNode()
_C.DATA.FLOAT = "float32"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_default_cfg() -> CfgNode:
    return _C.clone()


def _read_cfg_from_file_only(cfg_file: str) -> CfgNode:
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = CfgNode.load_cfg(f)
    return cfg


def get_nested_base(cfg: CfgNode):
    all_base = []
    while hasattr(cfg, "BASE") and cfg.BASE is not None:
        base, cfg = cfg.BASE, _read_cfg_from_file_only(cfg.BASE)
        all_base.append((base, cfg))
    return reversed(all_base)


def get_standard_float_dtype(float_type):
    mapping = {
        torch.float32: {"float", "float32", "f32", "single", "32"},
        torch.float64: {"double", "float64", "f64", "64"},
    }
    if isinstance(float_type, int):
        float_type = str(float_type)
    if isinstance(float_type, str):
        float_type = float_type.lower()
        for dtype, strs in mapping.items():
            if float_type in strs:
                return dtype
    return float_type


def set_default_dtype(dtype):
    torch.set_default_dtype(get_standard_float_dtype(dtype))


def apply_cfg(cfg: CfgNode):
    set_default_dtype(cfg.DATA.FLOAT)


def load_cfg(
    cfg_file: str,
    override: Optional[List[str]] = None,
) -> CfgNode:
    top_cfg = _read_cfg_from_file_only(cfg_file)
    all_base = get_nested_base(top_cfg)

    cfg = get_default_cfg()
    all_names: list[str] = []
    for _base_path, base_cfg in all_base:
        cfg.merge_from_other_cfg(base_cfg)
        if hasattr(base_cfg, "NAME"):
            all_names.append(base_cfg.NAME)
    cfg.merge_from_other_cfg(top_cfg)
    cfg.NAME = "_".join(all_names + [top_cfg.NAME])

    if hasattr(cfg, "BASE"):
        cfg.pop("BASE")

    if override is not None:
        cfg.merge_from_list(override)

    apply_cfg(cfg)
    cfg.freeze()
    return cfg
