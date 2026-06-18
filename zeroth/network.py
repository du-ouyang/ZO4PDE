"""Neural network architectures for the zeroth-order solver."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and dropout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        activation: str = "tanh",
        dropout_rate: float = 0.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

        self._initialize_weights(init_scale)

    def _initialize_weights(self, scale: float = 1.0):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            layer.weight.data *= scale
            nn.init.zeros_(layer.bias)
            if scale > 1.0:
                layer.bias.data += torch.randn_like(layer.bias) * (scale - 1.0)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.weight.data *= scale
        nn.init.zeros_(self.output_layer.bias)
        if scale > 1.0:
            self.output_layer.bias.data += torch.randn_like(self.output_layer.bias) * (scale - 1.0)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "relu":
            return F.relu(x)
        elif self.activation_type == "tanh":
            return torch.tanh(x)
        elif self.activation_type == "elu":
            return F.elu(x)
        elif self.activation_type == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(self._activate(layer(x)))
        return self.output_layer(x)


class PISGradNet(nn.Module):
    """DeepPicard-style HJB value network with built-in terminal-condition blending."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        dim: int,
        g0: Callable[[torch.Tensor], torch.Tensor],
        T: float = 1.0,
        channels: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim = dim
        self.channels = channels
        self.g0 = g0
        self.T = float(T)
        self.act = nn.ELU()

        self.timestep_phase = nn.Parameter(torch.zeros(1, channels))
        self.register_buffer("timestep_coeff", torch.linspace(0.1, 100.0, steps=channels).unsqueeze(0))

        self.t_encoder = nn.Sequential(
            nn.Linear(2 * channels, channels),
            self.act,
            nn.Linear(channels, channels),
        )

        smooth_layers: list[nn.Module] = [nn.Linear(2 * channels, channels)]
        for _ in range(num_layers):
            smooth_layers.append(self.act)
            smooth_layers.append(nn.Linear(channels, channels))
        smooth_layers.append(self.act)
        smooth_layers.append(nn.Linear(channels, dim))
        self.smooth_net = nn.Sequential(*smooth_layers)

        main_layers: list[nn.Module] = []
        in_dim = dim + channels
        for _ in range(num_layers):
            main_layers.append(nn.Linear(in_dim, hidden_size))
            main_layers.append(self.act)
            in_dim = hidden_size
        main_layers.append(nn.Linear(in_dim, dim))
        self.nn_module = nn.Sequential(*main_layers)

    def get_pis_timestep_embedding(self, lbd: torch.Tensor) -> torch.Tensor:
        arg = self.timestep_coeff * lbd + self.timestep_phase
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

    def smoothing_function(self, lbd: torch.Tensor) -> torch.Tensor:
        lbd_emb = self.get_pis_timestep_embedding(lbd)
        zero_emb = self.get_pis_timestep_embedding(torch.zeros_like(lbd))
        out_lbd = self.smooth_net(lbd_emb)
        out_zero = self.smooth_net(zero_emb)
        return out_lbd[..., 0:1] - out_zero[..., 0:1]

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        lbd = self.T - tx[..., 0:1]
        x = tx[..., 1:]

        smooth = self.smoothing_function(lbd)
        t_emb = self.t_encoder(self.get_pis_timestep_embedding(lbd))
        net_out = self.nn_module(torch.cat([t_emb, x], dim=-1))
        sp_out = torch.sum(net_out * x, dim=-1, keepdim=True)

        residual = self.g0(torch.exp(-0.5 * lbd) * x)
        return smooth * sp_out + (1.0 - smooth) * residual


def build_value_net(
    input_dim: int,
    hidden_size: int,
    num_layers: int,
    activation: str,
    dropout: float,
    init_scale: float,
    device: torch.device,
    *,
    value_net_type: str = "mlp",
    dim: Optional[int] = None,
    g0: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    T: float = 1.0,
) -> nn.Module:
    if value_net_type == "pisgradnet":
        if dim is None or g0 is None:
            raise ValueError("PISGradNet requires both dim and g0")
        return PISGradNet(hidden_size=hidden_size, num_layers=num_layers, dim=dim, g0=g0, T=T).to(device)

    return MLP(input_dim, hidden_size, num_layers, 1, activation, dropout, init_scale).to(device)


def build_three_nets(
    input_dim: int,
    d: int,
    hidden_size: int,
    num_layers: int,
    activation: str,
    dropout: float,
    init_scale: float,
    device: torch.device,
    train_hessian: bool = True,
    value_net_type: str = "mlp",
    g0: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    T: float = 1.0,
):
    """Build V, Grad, and optionally Hess networks.

    When *train_hessian* is False the returned ``Hess_net`` is ``None``;
    the Hessian is then obtained at eval time via autograd of ``Grad_net``.
    """
    V_net = build_value_net(
        input_dim,
        hidden_size,
        num_layers,
        activation,
        dropout,
        init_scale,
        device,
        value_net_type=value_net_type,
        dim=d,
        g0=g0,
        T=T,
    )
    Grad_net = MLP(input_dim, hidden_size + 2 * d, num_layers, d,
                   activation, dropout, init_scale).to(device)
    if train_hessian:
        Hess_net = MLP(input_dim, hidden_size + 2 * d * d, num_layers, d * d,
                       activation, dropout, init_scale).to(device)
    else:
        Hess_net = None
    return V_net, Grad_net, Hess_net
