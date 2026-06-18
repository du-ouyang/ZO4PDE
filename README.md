# Zeroth-Order Learning for Derivatives: Paper Examples

This repository contains the clean reproduction code for the three numerical
examples in Section 6 of the paper. It is intentionally small: generated
results, trained checkpoints, cached files, historical trial scripts, and
baseline output folders are not included.

The only binary files kept in the repository are small fixed parameter files
that define the paper benchmarks.

## Problem Setting

The paper studies parabolic PDEs of the form

```text
partial_t u(t, x) + L u(t, x) + f(t, x, u, grad u, Hess u) = 0,
u(T, x) = g(x),
```

where `L` is the generator of a simulated diffusion process `X`. The goal is to
learn not only the value function `u`, but also its first and second spatial
derivatives.

The core solver uses an approximate value-iteration/Picard scheme. At each
outer iteration, a nonlinear PDE is reduced to a linear Feynman-Kac regression
problem. Instead of training only a scalar value network and then differentiating
it by automatic differentiation, the code trains three networks:

- `V_net(t, x)` approximates the value `u(t, x)`.
- `Grad_net(t, x)` approximates `grad u(t, x)`.
- `Hess_net(t, x)` approximates `Hess u(t, x)`.

The derivative networks are trained using zeroth-order derivative (ZOD) targets:
function values are evaluated at perturbed initial states, and finite-difference
estimators provide stochastic targets for gradients and Hessians. In the
multi-point/common-noise setting, the same simulated noise is reused across
perturbed points, which reduces the variance of the derivative targets.

The first example is a one-dimensional linear PDE/regression diagnostic. It is
self-contained in `scripts/spiky_value/` because it compares NN-autodiff and ZOD
on a special spiky payoff. The second and third examples use the general
`zeroth` package and the `zeroth train ...` CLI.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- zeroth/
|-- scripts/
|   |-- spiky_value/
|   |-- hjb/
|   `-- fully_nonlinear/case1/
```

## File Guide

### Root Files

| File | Purpose |
|---|---|
| `.gitignore` | Keeps generated results, Python caches, logs, and trained model checkpoints out of git. |
| `README.md` | This guide: problem setting, file map, run commands, and output description. |
| `pyproject.toml` | Python package metadata and dependencies. It exposes the `zeroth` command-line entry point. |

### Core Solver: `zeroth/`

| File | Purpose |
|---|---|
| `zeroth/__init__.py` | Re-exports the main public objects: config loading, runner, evaluator, and plotting helper. |
| `zeroth/config.py` | Defines the default YACS configuration tree and loads YAML experiment files with command-line overrides. |
| `zeroth/equations.py` | Defines PDE/SDE benchmark classes, exact solutions, exact derivatives, terminal conditions, and nonlinear source terms. It contains the HJB/OU and fully nonlinear paper benchmarks. |
| `zeroth/network.py` | Defines neural network architectures: standard MLPs and the `PISGradNet` value-network ansatz used by the HJB example. |
| `zeroth/solver.py` | Implements the main three-network Picard training loop, value targets, ZOD derivative losses, derivative pretraining, warm starts, and checkpoint/evaluation calls. |
| `zeroth/evaluate.py` | Computes value, gradient, and Hessian errors against the exact benchmark solutions, including relative RMSEs. |
| `zeroth/data_saver.py` | Creates result folders and writes `config.yaml`, `error_history.csv`, checkpoints, and `training_report.txt`. |
| `zeroth/utils.py` | Provides a plotting helper for `error_history.csv`. |
| `zeroth/main.py` | Typer-based CLI. Main commands are `zeroth train <config.yaml>` and `zeroth plot <error_history.csv>`. |

### Linear PDE Diagnostic: `scripts/spiky_value/`

| File | Purpose |
|---|---|
| `scripts/spiky_value/README.md` | Short README specific to the spiky linear PDE diagnostic. |
| `scripts/spiky_value/run_spiky_value.py` | Self-contained script for Section 6.1. It builds the spiky payoff, trains the NN-autodiff baseline, trains the ZOD multi-point least-squares model, evaluates value/gradient/Hessian errors, and writes figures/tables. |
| `scripts/spiky_value/spiky_original_payoff.yaml` | Paper configuration for the spiky diagnostic: payoff variant, network size, batch sizes, learning rates, ZOD bandwidth, and output directory. |
| `scripts/spiky_value/replot_spiky_value.py` | Replots value, gradient, and Hessian curves from a saved `spiky_value_pointwise.csv` without rerunning training. |

### Semilinear HJB Example: `scripts/hjb/`

| File | Purpose |
|---|---|
| `scripts/hjb/paper_hjb_20d.yaml` | Section 6.2 configuration for the 20-dimensional semilinear HJB/OU benchmark. It uses the three-network solver, `PISGradNet` for the value network, ELU derivative networks, and multi-point ZOD training. |
| `scripts/hjb/mean_20d_ms=1.0_vs=2.0_5.pt` | Fixed Gaussian-mixture mean parameters for the HJB terminal condition. |
| `scripts/hjb/var_20d_ms=1.0_vs=2.0_5.pt` | Fixed Gaussian-mixture variance parameters for the HJB terminal condition. |
| `scripts/hjb/pi_20d_ms=1.0_vs=2.0_5.pt` | Fixed Gaussian-mixture mixture weights for the HJB terminal condition. |

### Fully Nonlinear Example: `scripts/fully_nonlinear/case1/`

| File | Purpose |
|---|---|
| `scripts/fully_nonlinear/case1/paper_fully_nonlinear_20d.yaml` | Section 6.3 configuration for the 20-dimensional fully nonlinear benchmark. It uses a three-network MLP model and the time-matched ZOD-m training budget reported in the paper. |
| `scripts/fully_nonlinear/case1/gbm_2nodes_w_20d.pt` | Fixed weight parameters defining the exact-solution ansatz for the fully nonlinear benchmark. |
| `scripts/fully_nonlinear/case1/gbm_2nodes_v_20d.pt` | Fixed node/vector parameters defining the exact-solution ansatz for the fully nonlinear benchmark. |

## Install

From this repository root:

```bash
pip install -e .
```

The code depends on PyTorch, NumPy, pandas, matplotlib, yacs, and Typer. A CUDA
GPU is strongly recommended for the HJB and fully nonlinear examples.

## Paper Example Map

| Paper section | Example | Main command |
|---|---|---|
| Section 6.1, A Linear PDE: Regression | 1D spiky value diagnostic | `python scripts/spiky_value/run_spiky_value.py --config scripts/spiky_value/spiky_original_payoff.yaml` |
| Section 6.2, A Semilinear PDE | 20D HJB/OU benchmark | `zeroth train scripts/hjb/paper_hjb_20d.yaml` |
| Section 6.3, A Fully Nonlinear PDE | 20D fully nonlinear benchmark | `zeroth train scripts/fully_nonlinear/case1/paper_fully_nonlinear_20d.yaml` |

The HJB and fully nonlinear examples in the paper are averaged over seeds 42,
43, and 44. To rerun other seeds, override `NAME` and `SEED` on the command
line:

```bash
zeroth train scripts/hjb/paper_hjb_20d.yaml NAME HJB_20D_paper_seed43 SEED 43
zeroth train scripts/fully_nonlinear/case1/paper_fully_nonlinear_20d.yaml NAME fully_nonlinear_20D_paper_seed43 SEED 43
```

## Outputs

The general `zeroth train` command writes outputs under `results/`:

- `config.yaml`: exact configuration used for the run.
- `error_history.csv`: value, gradient, and Hessian error history over Picard iterations.
- `training_report.txt`: final errors, timing, and configuration summary.
- `model_iter_*.pt` and `final_model.pt`: trained checkpoints.

The spiky diagnostic writes outputs under the directory specified by
`output.out_dir` in `spiky_original_payoff.yaml`:

- `config.yaml`: exact script configuration.
- `spiky_value_metrics.csv`: numeric RMSE and relative RMSE table.
- `spiky_value_pointwise.csv`: pointwise value/gradient/Hessian curves.
- `spiky_value_table.tex`: LaTeX table for the paper.
- `spiky_value_comparison.pdf`: value, gradient, and Hessian comparison figure.

All generated outputs are ignored by git.

## Notes

- The included `.pt` files are fixed benchmark parameters, not trained model
  outputs.
- Deep Picard baseline code and baseline output folders are not included here.
- This repository is meant to be the clean code artifact linked from the paper;
  the larger working directory contained many exploratory runs that are
  intentionally omitted.
