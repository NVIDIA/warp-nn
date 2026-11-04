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
Benchmark the tiled layers with fewer output features than the 2D tile width.

Usage from the repository root::

    python -m benchmarks.benchmark_narrow_layers --format markdown html json csv --output-dir benchmark-results
    python benchmarks/benchmark_narrow_layers.py --cuda-graph --modules "LSTM*"

The modules benchmark uses 256 features, wider than the tile. This benchmark covers the layers whose tile is fitted to
their output features (``Linear``, ``RNN``, ``GRU``, ``LSTM`` and their cells) at output widths from 1 to 64:
``Linear`` with 16 and 256 input features (the reduction size), the multi-layer recurrent layers with 2 input
features and 4 time steps (as Newton's actuator networks), and the cells with 64 input features. Like the modules
benchmark, it compares the current implementation with PyTorch and with a baseline git reference.
"""

from __future__ import annotations

import pathlib
import sys
import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "benchmarks"

from . import benchmark_modules
from ._specs import InputSpec, ModuleSpec, _call_lstm_cell


WIDTHS = (1, 4, 8, 12, 16, 24, 32, 48, 64)  # output features
LINEAR_INPUTS = (16, 256)  # input features of the Linear layers
CELL_INPUTS = 64  # input features of the cells
SEQUENCE_LENGTH = 4  # time steps of the multi-layer recurrent layers (with 2 input features)


def _layer(name: str, warp_factory, torch_factory, *inputs: InputSpec, **kwargs) -> ModuleSpec:
    return ModuleSpec(name=name, category="layers", warp=warp_factory, torch=torch_factory, inputs=inputs, **kwargs)


def _specs() -> tuple[ModuleSpec, ...]:
    specs = []
    for w in WIDTHS:
        for k in LINEAR_INPUTS:
            specs.append(
                _layer(
                    f"Linear {k}->{w}",
                    lambda nn, k=k, w=w: nn.Linear(k, w),
                    lambda k=k, w=w: torch.nn.Linear(k, w),
                    InputSpec((k,)),
                )
            )
        for name in ("RNN", "GRU", "LSTM"):
            specs.append(
                _layer(
                    f"{name} 2->{w}",
                    lambda nn, name=name, w=w: getattr(nn, name)(2, w),
                    lambda name=name, w=w: getattr(torch.nn, name)(2, w, batch_first=True),
                    InputSpec((SEQUENCE_LENGTH, 2)),
                )
            )
        for name in ("RNNCell", "GRUCell"):
            specs.append(
                _layer(
                    f"{name} {CELL_INPUTS}->{w}",
                    lambda nn, name=name, w=w: getattr(nn, name)(CELL_INPUTS, w),
                    lambda name=name, w=w: getattr(torch.nn, name)(CELL_INPUTS, w),
                    InputSpec((CELL_INPUTS,)),
                    InputSpec((w,)),
                )
            )
        specs.append(
            _layer(
                f"LSTMCell {CELL_INPUTS}->{w}",
                lambda nn, w=w: nn.LSTMCell(CELL_INPUTS, w),
                lambda w=w: torch.nn.LSTMCell(CELL_INPUTS, w),
                InputSpec((CELL_INPUTS,)),
                InputSpec((w,)),
                InputSpec((w,)),
                call=_call_lstm_cell,
            )
        )
    return tuple(specs)


SPECS = _specs()


def main(argv: list[str] | None = None) -> int:
    """Benchmark the narrow layers and write reports; return a failure status if a layer cannot be measured."""
    return benchmark_modules.main(argv, specs=SPECS, output_name="benchmark_narrow_layers", fail_on_errors=True)


if __name__ == "__main__":
    sys.exit(main())
