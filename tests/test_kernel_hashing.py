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

# Modules specialize their kernels by passing a Warp function to wp.tile_map via wp.static(): either a module-level
# function (e.g.: UnaryOp) or a closure function that captures Python values via wp.static() (e.g.: ELU).
# Since the kernels' source code is identical, Warp must hash the static function (and its static expressions)
# to tell the kernels apart. Otherwise, they silently share the same compiled code, both in the same process
# and across processes (via the kernel cache).

from typing import Any

import pytest

import os
import pathlib
import subprocess
import sys

import numpy as np
import warp as wp

from .utilities import is_device_available


TILE = 32


@wp.func
def _exp(x: Any):
    return wp.exp(x)


@wp.func
def _neg(x: Any):
    return -x


_FUNCTIONS = {"exp": (_exp, np.exp), "neg": (_neg, np.negative)}


def _create_kernel(variant: str, module: str | None):
    # variant: module-level function name, or value captured by a closure function
    if variant in _FUNCTIONS:
        function = _FUNCTIONS[variant][0]
    else:
        scale = float(variant)

        @wp.func
        def function(x: Any):
            return x.dtype(wp.static(scale)) * x

    @wp.kernel(module=module)
    def kernel(input: wp.array1d[Any], output: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(TILE),)
        offset = (i * wp.static(TILE),)
        tile = wp.tile_map(wp.static(function), wp.tile_load(input, shape=shape, offset=offset))
        wp.tile_store(output, tile, offset=offset)

    return wp.overload(kernel, [wp.array1d(dtype=wp.float32), wp.array1d(dtype=wp.float32)])


def _check_kernel(kernel, *, variant: str, device: str) -> None:
    array = np.linspace(-2.0, 2.0, 4 * TILE, dtype=np.float32)
    input = wp.array(array, device=device)
    output = wp.zeros_like(input)
    wp.launch_tiled(kernel, dim=(array.size // TILE,), inputs=[input], outputs=[output], device=device, block_dim=TILE)
    expected = _FUNCTIONS[variant][1](array) if variant in _FUNCTIONS else float(variant) * array
    assert np.allclose(output.numpy(), expected), f"Kernel '{variant}' ran another kernel's code"


def check_variant(variant: str, module: str | None, device: str) -> None:
    _check_kernel(_create_kernel(variant, module), variant=variant, device=device)


# test-specific parameters
@pytest.mark.parametrize("variants", [("exp", "neg"), ("0.5", "2.0")])
@pytest.mark.parametrize("module", [None, "unique"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_same_process(capsys, device, module, variants):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    first, second = variants
    kernel = _create_kernel(first, module)
    _check_kernel(kernel, variant=first, device=device)
    # create the second kernel after the first one has been loaded, and launch the first one again
    _check_kernel(_create_kernel(second, module), variant=second, device=device)
    _check_kernel(kernel, variant=first, device=device)


# test-specific parameters
@pytest.mark.parametrize("variants", [("exp", "neg"), ("0.5", "2.0")])
@pytest.mark.parametrize("module", [None, "unique"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_across_processes(capsys, tmp_path, device, module, variants):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # run each kernel in a new process, sharing the same (initially empty) kernel cache
    for variant in variants:
        code = f"from {__name__} import check_variant; check_variant({variant!r}, {module!r}, {device!r})"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=pathlib.Path(__file__).parents[1],
            env={**os.environ, "WARP_CACHE_PATH": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
