"""Zeroth-order PDE solver package."""

from zeroth.config import load_cfg, get_default_cfg
from zeroth.solver import ZerothOrderRunner
from zeroth.evaluate import Evaluator
from zeroth.utils import plot_error_history
