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

"""Benchmarked modules and their PyTorch counterparts.

The input sizes (other than the batch size) are representative of the policies/value functions of Isaac Lab tasks
(e.g. MLPs with hidden layers of 512, 256 and 128 units, recurrent layers with 256 hidden units, small images).
"""

from __future__ import annotations

from typing import Any, Callable

import dataclasses
from types import ModuleType
import torch

import numpy as np


CATEGORIES = ("activations", "layers", "operators")

FEATURES = 256  # number of features of the element-wise operations and the fully connected layers
CHANNELS = 16  # number of channels of the convolutional, pooling and per-channel normalization layers
LENGTH = 64  # signal length of the 1D convolutional, pooling and normalization layers
SIZE = 16  # height and width of the 2D convolutional and pooling layers
INPUT_SIZE = 64  # input size of the recurrent layers
HIDDEN_SIZE = 256  # hidden size of the recurrent layers
SEQUENCE_LENGTH = 16  # sequence length of the multi-layer recurrent layers
OBSERVATIONS = 48  # input size of the MLP
ACTIONS = 12  # output size of the MLP


@dataclasses.dataclass(frozen=True)
class InputSpec:
    """Specification of a module input, with shape ``(batch_size, *shape)``."""

    shape: tuple[int, ...]
    """Shape of the input, excluding the (leading) batch dimension."""

    low: float = -1.0
    """Lower bound of the (uniformly sampled) input values."""

    high: float = 1.0
    """Upper bound of the (uniformly sampled) input values."""

    integer: bool = False
    """Whether the input is a 32-bit integer array (rather than a 32-bit floating point array)."""

    def sample(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample the input values.

        :param batch_size: The batch size.
        :param rng: The random number generator.

        :return: The sampled input array.
        """
        shape = (batch_size, *self.shape)
        if self.integer:
            return rng.integers(-1000, 1000, size=shape, dtype=np.int32)
        return rng.uniform(self.low, self.high, size=shape).astype(np.float32)


def _call(module: Any, *inputs: Any) -> Any:
    return module(*inputs)


@dataclasses.dataclass(frozen=True)
class ModuleSpec:
    """Specification of a benchmarked module and its PyTorch counterpart."""

    name: str
    """Name of the benchmark (e.g. the module name)."""

    category: str
    """Category of the module (one of :py:data:`CATEGORIES`)."""

    warp: Callable[[ModuleType], Any]
    """Factory of the warp-nn module, given the ``nn`` namespace of the implementation (current or baseline)."""

    torch: Callable[[], Any]
    """Factory of the PyTorch counterpart (a ``torch.nn.Module`` or a function)."""

    inputs: tuple[InputSpec, ...] = (InputSpec((FEATURES,)),)
    """Specifications of the inputs."""

    call: Callable[..., Any] = _call
    """Function that calls the module (warp-nn or PyTorch) with the inputs, and returns its output(s)."""

    differentiable: bool = True
    """Whether the module is differentiable (i.e. whether its backward pass is benchmarked)."""

    def sample_inputs(self, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Sample the inputs.

        :param batch_size: The batch size.
        :param rng: The random number generator.

        :return: The sampled input arrays.
        """
        return [spec.sample(batch_size, rng) for spec in self.inputs]


def _activation(name: str, torch_factory: Callable[[], Any], **kwargs) -> ModuleSpec:
    return ModuleSpec(
        name=name, category="activations", warp=lambda nn: getattr(nn, name)(**kwargs), torch=torch_factory
    )


def _operator(name: str, torch_function: Callable[..., Any], *inputs: InputSpec, **kwargs) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        category="operators",
        warp=lambda nn: getattr(nn, name)(**kwargs),
        torch=lambda: torch_function,
        inputs=inputs or (InputSpec((FEATURES,)),),
    )


def _bitwise(name: str, torch_function: Callable[..., Any], num_inputs: int) -> ModuleSpec:
    return ModuleSpec(
        name=name,
        category="operators",
        warp=lambda nn: getattr(nn, name)(),
        torch=lambda: torch_function,
        inputs=(InputSpec((FEATURES,), integer=True),) * num_inputs,
        differentiable=False,
    )


def _layer(
    name: str,
    warp_factory: Callable[[ModuleType], Any],
    torch_factory: Callable[[], Any],
    *inputs: InputSpec,
    call: Callable[..., Any] = _call,
) -> ModuleSpec:
    return ModuleSpec(name=name, category="layers", warp=warp_factory, torch=torch_factory, inputs=inputs, call=call)


def _call_lstm_cell(module: Any, input: Any, hidden: Any, cell: Any) -> Any:
    return module(input, (hidden, cell))


def _torch_prelu(x: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    # element-wise slope (ONNX PRelu semantics), which torch.nn.functional.prelu does not support
    return torch.where(x >= 0, x, slope * x)


def _torch_clip(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -0.5, 0.5)


def _warp_mlp(nn: ModuleType) -> Any:
    return nn.Sequential(
        nn.Linear(OBSERVATIONS, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, ACTIONS),
    )


def _torch_mlp() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(OBSERVATIONS, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, ACTIONS),
    )


_VECTOR = InputSpec((FEATURES,))
_SIGNAL = InputSpec((CHANNELS, LENGTH))
_IMAGE = InputSpec((CHANNELS, SIZE, SIZE))
_POSITIVE = InputSpec((FEATURES,), low=0.1, high=2.0)
_UNIT = InputSpec((FEATURES,), low=-0.9, high=0.9)

SPECS: tuple[ModuleSpec, ...] = (
    # activations
    _activation("CELU", torch.nn.CELU),
    _activation("ELU", torch.nn.ELU),
    _activation("GELU", torch.nn.GELU),
    _activation("HardSigmoid", torch.nn.Hardsigmoid),
    _activation("HardSwish", torch.nn.Hardswish),
    _activation("LeakyReLU", torch.nn.LeakyReLU),
    _activation("LogSoftmax", lambda: torch.nn.LogSoftmax(dim=-1), dim=-1),
    _activation("Mish", torch.nn.Mish),
    ModuleSpec(
        name="PReLU",
        category="activations",
        warp=lambda nn: nn.PReLU(),
        torch=lambda: _torch_prelu,
        inputs=(_VECTOR, InputSpec((FEATURES,), low=0.0, high=0.5)),
    ),
    _activation("ReLU", torch.nn.ReLU),
    _activation("SELU", torch.nn.SELU),
    _activation("Shrink", lambda: torch.nn.Softshrink(0.5), lambd=0.5, bias=0.5),
    _activation("Sigmoid", torch.nn.Sigmoid),
    _activation("Softmax", lambda: torch.nn.Softmax(dim=-1), dim=-1),
    _activation("Softplus", torch.nn.Softplus),
    _activation("Softsign", torch.nn.Softsign),
    _activation("Swish", torch.nn.SiLU),
    _activation("Threshold", lambda: torch.nn.Threshold(0.1, 20.0), threshold=0.1, value=20.0),
    # layers
    _layer("AvgPool1D", lambda nn: nn.AvgPool1D(2), lambda: torch.nn.AvgPool1d(2), _SIGNAL),
    _layer("AvgPool2D", lambda nn: nn.AvgPool2D(2), lambda: torch.nn.AvgPool2d(2), _IMAGE),
    _layer("BatchNorm", lambda nn: nn.BatchNorm(FEATURES), lambda: torch.nn.BatchNorm1d(FEATURES), _VECTOR),
    _layer(
        "Conv1D",
        lambda nn: nn.Conv1D(CHANNELS, 2 * CHANNELS, 3, padding=1),
        lambda: torch.nn.Conv1d(CHANNELS, 2 * CHANNELS, 3, padding=1),
        _SIGNAL,
    ),
    _layer(
        "Conv2D",
        lambda nn: nn.Conv2D(3, CHANNELS, 3, padding=1),
        lambda: torch.nn.Conv2d(3, CHANNELS, 3, padding=1),
        InputSpec((3, SIZE, SIZE)),
    ),
    _layer("Dropout", lambda nn: nn.Dropout(0.5), lambda: torch.nn.Dropout(0.5), _VECTOR),
    _layer("Flatten", lambda nn: nn.Flatten(), torch.nn.Flatten, InputSpec((CHANNELS, FEATURES // CHANNELS))),
    _layer("GlobalAvgPool", lambda nn: nn.GlobalAvgPool(), lambda: torch.nn.AdaptiveAvgPool2d(1), _IMAGE),
    _layer("GlobalMaxPool", lambda nn: nn.GlobalMaxPool(), lambda: torch.nn.AdaptiveMaxPool2d(1), _IMAGE),
    _layer("GroupNorm", lambda nn: nn.GroupNorm(4, CHANNELS), lambda: torch.nn.GroupNorm(4, CHANNELS), _SIGNAL),
    _layer(
        "GRU",
        lambda nn: nn.GRU(INPUT_SIZE, HIDDEN_SIZE),
        lambda: torch.nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=True),
        InputSpec((SEQUENCE_LENGTH, INPUT_SIZE)),
    ),
    _layer(
        "GRUCell",
        lambda nn: nn.GRUCell(INPUT_SIZE, HIDDEN_SIZE),
        lambda: torch.nn.GRUCell(INPUT_SIZE, HIDDEN_SIZE),
        InputSpec((INPUT_SIZE,)),
        InputSpec((HIDDEN_SIZE,)),
    ),
    _layer("Identity", lambda nn: nn.Identity(), torch.nn.Identity, _VECTOR),
    _layer("InstanceNorm", lambda nn: nn.InstanceNorm(CHANNELS), lambda: torch.nn.InstanceNorm1d(CHANNELS), _SIGNAL),
    _layer("LayerNorm", lambda nn: nn.LayerNorm(FEATURES), lambda: torch.nn.LayerNorm(FEATURES), _VECTOR),
    _layer("Linear", lambda nn: nn.Linear(FEATURES, FEATURES), lambda: torch.nn.Linear(FEATURES, FEATURES), _VECTOR),
    _layer(
        "LSTM",
        lambda nn: nn.LSTM(INPUT_SIZE, HIDDEN_SIZE),
        lambda: torch.nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True),
        InputSpec((SEQUENCE_LENGTH, INPUT_SIZE)),
    ),
    _layer(
        "LSTMCell",
        lambda nn: nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE),
        lambda: torch.nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE),
        InputSpec((INPUT_SIZE,)),
        InputSpec((HIDDEN_SIZE,)),
        InputSpec((HIDDEN_SIZE,)),
        call=_call_lstm_cell,
    ),
    _layer("MaxPool1D", lambda nn: nn.MaxPool1D(2), lambda: torch.nn.MaxPool1d(2), _SIGNAL),
    _layer("MaxPool2D", lambda nn: nn.MaxPool2D(2), lambda: torch.nn.MaxPool2d(2), _IMAGE),
    _layer("RMSNorm", lambda nn: nn.RMSNorm(FEATURES), lambda: torch.nn.RMSNorm(FEATURES), _VECTOR),
    _layer(
        "RNN",
        lambda nn: nn.RNN(INPUT_SIZE, HIDDEN_SIZE),
        lambda: torch.nn.RNN(INPUT_SIZE, HIDDEN_SIZE, batch_first=True),
        InputSpec((SEQUENCE_LENGTH, INPUT_SIZE)),
    ),
    _layer(
        "RNNCell",
        lambda nn: nn.RNNCell(INPUT_SIZE, HIDDEN_SIZE),
        lambda: torch.nn.RNNCell(INPUT_SIZE, HIDDEN_SIZE),
        InputSpec((INPUT_SIZE,)),
        InputSpec((HIDDEN_SIZE,)),
    ),
    _layer("Sequential (MLP)", _warp_mlp, _torch_mlp, InputSpec((OBSERVATIONS,))),
    # operators
    _operator("Abs", torch.abs),
    _operator("Acos", torch.acos, _UNIT),
    _operator("Acosh", torch.acosh, InputSpec((FEATURES,), low=1.1, high=3.0)),
    _operator("Add", torch.add, _VECTOR, _VECTOR),
    _operator("Asin", torch.asin, _UNIT),
    _operator("Asinh", torch.asinh),
    _operator("Atan", torch.atan),
    _operator("Atanh", torch.atanh, _UNIT),
    _bitwise("BitwiseAnd", torch.bitwise_and, 2),
    _bitwise("BitwiseNot", torch.bitwise_not, 1),
    _bitwise("BitwiseOr", torch.bitwise_or, 2),
    _bitwise("BitwiseXor", torch.bitwise_xor, 2),
    _operator("Ceil", torch.ceil),
    _operator("Clip", _torch_clip, min_val=-0.5, max_val=0.5),
    _operator("Cos", torch.cos),
    _operator("Cosh", torch.cosh),
    _operator("Div", torch.div, _VECTOR, InputSpec((FEATURES,), low=0.5, high=2.0)),
    _operator("Erf", torch.erf),
    _operator("Exp", torch.exp),
    _operator("Floor", torch.floor),
    _operator("Log", torch.log, _POSITIVE),
    _operator("Max", torch.maximum, _VECTOR, _VECTOR),
    _operator("Min", torch.minimum, _VECTOR, _VECTOR),
    _operator("Mul", torch.mul, _VECTOR, _VECTOR),
    _operator("Neg", torch.neg),
    _operator("Pow", torch.pow, InputSpec((FEATURES,), low=0.5, high=2.0), _VECTOR),
    _operator("Reciprocal", torch.reciprocal, InputSpec((FEATURES,), low=0.5, high=2.0)),
    _operator("Round", torch.round),
    _operator("Sign", torch.sign),
    _operator("Sin", torch.sin),
    _operator("Sinh", torch.sinh),
    _operator("Sqrt", torch.sqrt, _POSITIVE),
    _operator("Sub", torch.sub, _VECTOR, _VECTOR),
    _operator("Tan", torch.tan, _UNIT),
    _operator("Tanh", torch.tanh),
)
