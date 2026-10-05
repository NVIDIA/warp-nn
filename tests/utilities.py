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

from typing import Literal

import math
import os
import torch

import numpy as np
import warp as wp

from warp_nn.utils import logger


def is_device_available(device: str) -> bool:
    try:
        wp.get_device(device)
    except Exception as e:
        logger.warning(f"Device '{device}' is not available: {e}")
        return False
    return True


def is_running_on_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") is not None


def parse_dtype(framework: str, dtype: type | None) -> type:
    mapping = {
        "numpy": {
            None: np.float32,
            wp.float16: np.float16,
            wp.float32: np.float32,
            wp.float64: np.float64,
        },
        "torch": {
            None: torch.float32,
            wp.float16: torch.float16,
            wp.float32: torch.float32,
            wp.float64: torch.float64,
        },
        "warp": {
            None: wp.float32,
            wp.float16: wp.float16,
            wp.float32: wp.float32,
            wp.float64: wp.float64,
        },
    }
    return mapping[framework][dtype]


def sample_array(shape: tuple[int, ...], dtype: type | None = None) -> np.ndarray:
    return (2.0 * np.random.rand(*shape) - 1.0).astype(parse_dtype("numpy", dtype))


def init_parameters(parameters) -> None:
    for parameter in parameters:
        shape = tuple(parameter.shape)
        array = np.linspace(-1.0, 1.0, math.prod(shape)).reshape(shape)
        if isinstance(parameter, wp.array):
            wp.copy(parameter, wp.array(array, device=parameter.device, dtype=parameter.dtype))
        elif isinstance(parameter, torch.Tensor):
            with torch.no_grad():
                parameter.copy_(torch.tensor(array, device=parameter.device, dtype=parameter.dtype))
        else:
            raise ValueError(f"Unsupported parameter type: {type(parameter)}")


def check_arrays(
    a,
    b,
    /,
    *,
    flatten: bool = False,
    rtol: float = 0.0,
    atol: float = 1e-03,
    test: Literal["all-close", "equal", "not-equal"] = "all-close",
    msg: str = "",
):
    def _prepare(data):
        data = data if isinstance(data, (list, tuple)) else [data]
        data = [d.numpy() if isinstance(d, wp.array) else d.detach().cpu().numpy() for d in data]
        wp.synchronize()
        if flatten:
            data = [d.flatten() for d in data]
        return data

    wp.synchronize()
    assert len(a) == len(b), f"Unexpected length: expected {len(a)}, got {len(b)}"
    for i, (a, b) in enumerate(zip(_prepare(a), _prepare(b))):
        # shape
        assert a.shape == b.shape, f"Unexpected shape (at index {i}): expected {a.shape}, got {b.shape}"
        # value
        diff = a - b
        abs_diff = np.abs(diff)
        stats = (
            f"{np.mean(abs_diff)} +- {np.std(abs_diff)}, "
            f"limits: [{np.min(abs_diff)}, {np.max(abs_diff)}], sum: {np.sum(abs_diff)}"
        )
        if test == "all-close":
            # |b - a| <= atol + rtol * |a|, where the first array (a) is the reference
            assert np.allclose(b, a, rtol=rtol, atol=atol), f"[all-close] Failed (at index {i}): {msg}: {stats}"
        elif test == "equal":
            assert np.array_equiv(diff, 0.0), f"[equal] Failed (at index {i}): {msg}: {stats}"
        elif test == "not-equal":
            assert not np.array_equiv(diff, 0.0), f"[not-equal] Failed (at index {i}): {msg}: {stats}"
        else:
            raise ValueError(f"Invalid test: {test}")


def backward(tape: wp.Tape, torch_outputs: list[torch.Tensor], warp_outputs: list[wp.array]) -> None:
    """Backpropagate the same random upstream gradients through the PyTorch and Warp outputs.

    The upstream gradients are non-uniform (rather than those of the sum of the outputs, i.e., ones),
    so that the checked gradients depend on how each output element propagates its own gradient
    (e.g., a backward pass that ignores the incoming gradients, or that mixes up the output indices,
    yields the same gradients as the reference for the sum of the outputs, but not for a weighted sum).
    """
    assert len(torch_outputs) == len(warp_outputs) > 0, "Expected the same (non-zero) number of outputs"
    # sampled in the precision of the Warp outputs, so that both frameworks use exactly the same values
    weights = [sample_array(tuple(output.shape)).astype(wp.dtype_to_numpy(output.dtype)) for output in warp_outputs]
    loss = sum(
        (output * torch.tensor(w, device=output.device, dtype=output.dtype)).sum()
        for output, w in zip(torch_outputs, weights, strict=True)
    )
    loss.backward()
    tape.backward(
        grads={
            output: wp.array(w, device=output.device, dtype=output.dtype)
            for output, w in zip(warp_outputs, weights, strict=True)
        }
    )
