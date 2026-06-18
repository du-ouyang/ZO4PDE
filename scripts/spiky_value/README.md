# Spiky Value Linear PDE Diagnostic

This folder contains the Section 6.1 linear PDE/regression diagnostic from the
paper. It compares learning a value network and differentiating it by autograd
against directly learning value, gradient, and Hessian networks with multi-point
zeroth-order targets.

Run from the repository root:

```bash
python scripts/spiky_value/run_spiky_value.py --config scripts/spiky_value/spiky_original_payoff.yaml
```

The paper configuration is `spiky_original_payoff.yaml`. It uses:

- the original `spiky` payoff,
- payoff-sample value losses for both methods,
- `B = 32768` for NN-autodiff value training,
- `B = 16384` for ZOD online multi-point least squares,
- ZOD bandwidth `eps = 0.01`.

For a quick smoke test:

```bash
python scripts/spiky_value/run_spiky_value.py --config scripts/spiky_value/spiky_original_payoff.yaml --value_steps 1000 --online_steps 300 --out_dir scripts/spiky_value/results_smoke
```

Each full run writes:

- `config.yaml`
- `spiky_value_comparison.pdf`
- `spiky_value_metrics.csv`
- `spiky_value_pointwise.csv`
- `spiky_value_table.tex`

Use `replot_spiky_value.py` to redraw the figure from a saved
`spiky_value_pointwise.csv`.
