"""CLI entry point for the paper-example zeroth-order solver."""

from pathlib import Path

import typer

from zeroth import load_cfg, ZerothOrderRunner, plot_error_history

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(configfile: str, ctx: typer.Context):
    """Train the zeroth-order PDE solver from a YAML config."""
    cfg_path = Path(configfile)
    assert cfg_path.exists(), f"Config file {configfile} not found"
    override = [arg.lstrip("-") for arg in ctx.args]
    cfg = load_cfg(configfile, override)
    runner = ZerothOrderRunner(cfg, config_path=str(cfg_path.resolve()))
    runner.run()


@app.command()
def plot(csv_path: str, relative: bool = True):
    """Plot error history from a saved CSV."""
    p = Path(csv_path)
    if not p.exists():
        # Strip a leading "program/" or "program\" segment for compatibility
        # with paths copied from the original repository.
        parts = p.parts
        if parts and parts[0].lower() == "program":
            p = Path(*parts[1:])
    plot_error_history(str(p), relative=relative)


if __name__ == "__main__":
    app()
