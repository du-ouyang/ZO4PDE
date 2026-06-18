# Zeroth-Order Learning for Derivatives: Paper Examples

This is a clean, GitHub-ready subset of the original experimental repository.
It keeps only the code needed for the three numerical examples that appear in
Section 6 of the paper, plus the small fixed parameter files that define those
benchmarks.

No generated result folders, checkpoints, cached files, or historical trial
outputs are included.

## Layout

```text
zeroth/
  Core three-network zeroth-order Picard solver.
scripts/
  spiky_value/
    Linear PDE diagnostic from Section 6.1.
  hjb/
    Semilinear 20D HJB/OU benchmark from Section 6.2.
  fully_nonlinear/case1/
    20D fully nonlinear benchmark from Section 6.3.
```

## Install

From this repository root:

```bash
pip install -e .
```

The project uses PyTorch, NumPy, pandas, matplotlib, yacs, and Typer. A CUDA
GPU is strongly recommended for the HJB and fully nonlinear runs.

## Paper Example Map

| Paper section | Example | Main command |
|---|---|---|
| Section 6.1, A Linear PDE: Regression | 1D spiky value diagnostic | `python scripts/spiky_value/run_spiky_value.py --config scripts/spiky_value/spiky_original_payoff.yaml` |
| Section 6.2, A Semilinear PDE | 20D HJB/OU benchmark | `zeroth train scripts/hjb/paper_hjb_20d.yaml` |
| Section 6.3, A Fully Nonlinear PDE | 20D fully nonlinear benchmark | `zeroth train scripts/fully_nonlinear/case1/paper_fully_nonlinear_20d.yaml` |

The HJB and fully nonlinear examples were reported over seeds 42, 43, and 44.
To reproduce a seed sweep, rerun the same config with command-line overrides:

```bash
zeroth train scripts/hjb/paper_hjb_20d.yaml NAME HJB_20D_paper_seed43 SEED 43
zeroth train scripts/fully_nonlinear/case1/paper_fully_nonlinear_20d.yaml NAME fully_nonlinear_20D_paper_seed43 SEED 43
```

## Outputs

Training outputs are written under `results/` by default. That directory is
ignored by git. The spiky diagnostic writes its outputs under
`scripts/spiky_value/results_spiky_zod_bs16384_eps001/`, also ignored by git.

Each `zeroth train` run writes:

- `config.yaml`
- `error_history.csv`
- `training_report.txt`
- model checkpoints

The spiky script writes:

- `config.yaml`
- `spiky_value_metrics.csv`
- `spiky_value_pointwise.csv`
- `spiky_value_table.tex`
- `spiky_value_comparison.pdf`

## Notes

- The `.pt` files in `scripts/hjb/` and `scripts/fully_nonlinear/case1/` are
  fixed benchmark parameters, not trained model outputs.
- Deep Picard baseline outputs and historical post-processing files are not
  included in this clean repository.
