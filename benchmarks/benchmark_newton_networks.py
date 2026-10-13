# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark Newton actuator network inference in Warp-NN against matching PyTorch networks.

Usage from the repository root::

    python -m benchmarks.benchmark_newton_networks --format markdown html json csv --output-dir benchmark-results
    python benchmarks/benchmark_newton_networks.py --cuda-graph

The network definitions follow Newton's checkpoint contracts. Newton accepts checkpoint-defined architectures;
the MLP here uses three 32-unit ELU layers, and the LSTM uses one 8-unit layer with a linear effort decoder.
The three-32-unit MLPs are representative checkpoints, not fixed Newton defaults. The additional single-linear
MLP matches ``_build_mlp_onnx``. GRU configurations use the four normalized features and four-unit hidden size of
``TestDriveNeuralGRU``; their stacked cells and optional Tanh head follow ``DriveNeuralGRU.finalize/compute``.
Newton applies the optional GRU output scale in its effort-writing kernel, outside the timed network.
No Newton installation or checkpoint download is needed. Batch size counts actuators, not environments.
Only network inference is timed: features are already scaled/stacked, and joint indexing, history updates and
effort scaling are excluded. Each LSTM evaluation receives preallocated zero states and processes one time step.
GRU evaluations likewise receive preallocated zero states and process one time step.
The recurrent output states are computed but are not fed back during timing.

Newton's ``DriveNeuralMLP`` and ``DriveNeuralLSTM`` evaluate one network per actuator (each actuator is a batch
element) from two features: the scaled position error ``(target_q - q) * pos_scale`` and the scaled joint velocity
``qd * vel_scale``. The network output is a single value per actuator that is multiplied by ``effort_scale`` to get
the joint effort.

- :py:class:`ActuatorMLP` maps the (optionally history-stacked) features to the effort:
  ``(batch_size, input_size) -> Linear -> activation -> ... -> Linear -> (batch_size, 1)``.
- :py:class:`ActuatorLSTM` maps a feature sequence to the effort while carrying hidden and cell states across calls:
  ``(batch_size, sequence_length, 2) -> LSTM -> last time step -> Linear -> (batch_size, 1)``.
- :py:class:`ActuatorGRU` follows Newton's single-step cell loop with one hidden-state array per layer:
  ``(batch_size, 4) -> GRUCell(s) -> Linear -> optional Tanh -> (batch_size, 1)``.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable, Sequence
import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "benchmarks"

import warp as wp

import warp_nn.nn as nn

from . import benchmark_modules
from ._specs import InputSpec, ModuleSpec


class ActuatorMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_sizes: Sequence[int] = (32, 32, 32),
        *,
        activation: Callable[[], nn.Module] = nn.ELU,
    ):
        """Multilayer perceptron (MLP) actuator network.

        The input concatenates the position error block and the joint velocity block (in the order given by
        Newton's ``input_order`` metadata), each with one column per history index in ``input_idx``.
        So ``input_size`` is ``2 * len(input_idx)``, i.e. ``2`` when no history is used.

        :param input_size: The number of input features.
        :param hidden_sizes: The number of features of each hidden layer. If empty, the network is a single
            linear layer.
        :param activation: Factory of the activation module applied after each hidden layer.

        :raises ValueError: If the input size is not a positive even number.
        """
        super().__init__()
        if input_size < 2 or input_size % 2:
            raise ValueError(f"The input size must be a positive even number (pos/vel blocks), got {input_size}")
        layers = []
        in_features = input_size
        for out_features in hidden_sizes:
            layers += [nn.Linear(in_features, out_features), activation()]
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)

        # register sub-modules and parameters
        super().__post_init__()

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the network.

        :param input: The input array, with shape ``(batch_size, input_size)``.

        :return: The (unscaled) effort array, with shape ``(batch_size, 1)``.
        """
        return self.mlp(input)


class ActuatorLSTM(nn.Module):
    def __init__(self, hidden_size: int = 8, *, num_layers: int = 1):
        """Long Short-Term Memory (LSTM) actuator network.

        The input features are the position error and the joint velocity of the current time step, so the history
        is carried by the hidden and cell states (with shapes ``(num_layers, batch_size, hidden_size)``) instead of
        an explicit input window. The effort is decoded from the last layer's hidden state at the last time step.

        :param hidden_size: The number of hidden features.
        :param num_layers: The number of stacked recurrent layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(2, hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, 1)

        # register sub-modules and parameters
        super().__post_init__()

    def __call__(
        self, input: wp.array, hidden: tuple[wp.array, wp.array] | None = None
    ) -> tuple[wp.array, tuple[wp.array, wp.array]]:
        """Forward pass of the network.

        .. note::

            The returned arrays are cached by the modules and overwritten by the next forward pass with the same
            input shape. They can be passed back as ``hidden`` to step the network, but must be copied if their
            values are needed after the next forward pass.

        :param input: The input sequence array, with shape ``(batch_size, sequence_length, 2)``.
        :param hidden: A tuple of the initial hidden state and cell state arrays, both with shapes
            ``(num_layers, batch_size, hidden_size)``. If not given, they default to zeros.

        :return: A tuple of the (unscaled) effort array, with shape ``(batch_size, 1)``, and a tuple of the final
            hidden state and cell state arrays, both with shapes ``(num_layers, batch_size, hidden_size)``.
        """
        output, hidden = self.lstm(input, hidden)
        return self.decoder(output[:, -1]), hidden


class TorchActuatorLSTM(torch.nn.Module):
    """Newton's ``_LSTMNet`` checkpoint architecture, with an optional zero initial state."""

    def __init__(self, hidden_size: int = 8, *, num_layers: int = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_size, 1)

    def forward(self, input, hidden=None):
        output, hidden = self.lstm(input, hidden)
        return self.decoder(output[:, -1, :]), hidden


class ActuatorGRU(nn.Module):
    """Single-step GRU cells and effort head reconstructed by Newton's ``DriveNeuralGRU``."""

    def __init__(self, input_size: int = 4, hidden_size: int = 4, *, num_layers: int = 1, apply_tanh: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.layers = []
        for index in range(num_layers):
            layer = nn.GRUCell(input_size if index == 0 else hidden_size, hidden_size)
            self.register_module(f"layer_{index}", layer)
            self.layers.append(layer)
        self.decoder = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh() if apply_tanh else nn.Identity()
        super().__post_init__()

    def __call__(self, input: wp.array, hidden: tuple[wp.array, ...]):
        if len(hidden) != self.num_layers:
            raise ValueError("Expected one hidden state per GRU layer")
        states = []
        for layer, state in zip(self.layers, hidden):
            input = layer(input, state)
            states.append(input)
        return self.activation(self.decoder(input)), tuple(states)


class TorchActuatorGRU(torch.nn.Module):
    """Torch counterpart of Newton's stacked single-step GRUCell evaluation."""

    def __init__(self, input_size: int = 4, hidden_size: int = 4, *, num_layers: int = 1, apply_tanh: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            torch.nn.GRUCell(input_size if index == 0 else hidden_size, hidden_size) for index in range(num_layers)
        )
        self.decoder = torch.nn.Linear(hidden_size, 1)
        self.activation = torch.nn.Tanh() if apply_tanh else torch.nn.Identity()

    def forward(self, input, hidden):
        if len(hidden) != self.num_layers:
            raise ValueError("Expected one hidden state per GRU layer")
        states = []
        for layer, state in zip(self.layers, hidden):
            input = layer(input, state)
            states.append(input)
        return self.activation(self.decoder(input)), tuple(states)


def torch_mlp(input_size: int = 2) -> torch.nn.Module:
    """MLP checkpoint architecture accepted by Newton's ``DriveNeuralMLP``."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 1),
    )


def _call_lstm(module, input, hidden, cell):
    # InputSpec stores the batch dimension first; Newton's recurrent states have layers first.
    shape = (1, input.shape[0], 8)
    return module(input, (hidden.reshape(shape), cell.reshape(shape)))


def _call_gru(module, input, hidden):
    return module(input, tuple(hidden[:, index] for index in range(module.num_layers)))


SPECS = (
    ModuleSpec(
        name="Newton MLP",
        category="networks",
        warp=lambda nn: ActuatorMLP(),
        torch=torch_mlp,
        inputs=(InputSpec((2,)),),
        differentiable=False,
    ),
    ModuleSpec(
        name="Newton MLP (history)",
        category="networks",
        warp=lambda nn: ActuatorMLP(input_size=6),
        torch=lambda: torch_mlp(6),
        inputs=(InputSpec((6,)),),
        differentiable=False,
    ),
    ModuleSpec(
        name="Newton LSTM",
        category="networks",
        warp=lambda nn: ActuatorLSTM(),
        torch=TorchActuatorLSTM,
        inputs=(InputSpec((1, 2)), InputSpec((8,), low=0.0, high=0.0), InputSpec((8,), low=0.0, high=0.0)),
        call=_call_lstm,
        differentiable=False,
    ),
    ModuleSpec(
        name="Newton MLP (linear reference)",
        category="networks",
        warp=lambda nn: ActuatorMLP(hidden_sizes=()),
        torch=lambda: torch.nn.Sequential(torch.nn.Linear(2, 1)),
        inputs=(InputSpec((2,)),),
        differentiable=False,
    ),
    ModuleSpec(
        name="Newton GRU",
        category="networks",
        warp=lambda nn: ActuatorGRU(),
        torch=TorchActuatorGRU,
        inputs=(InputSpec((4,)), InputSpec((1, 4), low=0.0, high=0.0)),
        call=_call_gru,
        differentiable=False,
    ),
    ModuleSpec(
        name="Newton GRU (stacked, Tanh head)",
        category="networks",
        warp=lambda nn: ActuatorGRU(num_layers=2, apply_tanh=True),
        torch=lambda: TorchActuatorGRU(num_layers=2, apply_tanh=True),
        inputs=(InputSpec((4,)), InputSpec((2, 4), low=0.0, high=0.0)),
        call=_call_gru,
        differentiable=False,
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Benchmark inference and write reports; return a failure status if any network cannot be measured."""
    return benchmark_modules.main(
        argv,
        specs=SPECS,
        output_name="benchmark_newton_networks",
        no_baseline=True,
        fail_on_errors=True,
    )


if __name__ == "__main__":
    sys.exit(main())
