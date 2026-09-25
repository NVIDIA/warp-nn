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

import pytest

import torch

import numpy as np
import warp as wp

import warp_nn.nn as nn

from ... import utilities
from .common import check_forward, check_gradients, check_initialize_parameters, check_requires_grad


_TORCH_MODULES = {2: torch.nn.BatchNorm1d, 3: torch.nn.BatchNorm1d, 4: torch.nn.BatchNorm2d}
_SHAPES = [(8, 5), (4, 5, 7), (3, 5, 4, 6), (2, 5, 1)]


def _train_both(warp_module, torch_module, *, shape, device, steps: int = 3):
    """Run some forward passes in training mode, to update the running statistics."""
    for _ in range(steps):
        array = utilities.sample_array(shape) + np.random.rand(1, shape[1], *([1] * (len(shape) - 2)))
        warp_module(wp.array(array.astype(np.float32), device=device))
        torch_module(torch.tensor(array.astype(np.float32), device=device))


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(capsys, device, dtype, shape, track_running_stats, affine):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    kwargs = {"eps": 1e-3, "affine": affine, "track_running_stats": track_running_stats}
    check_forward(
        warp_module=nn.BatchNorm(shape[1], **kwargs),
        torch_module=_TORCH_MODULES[len(shape)](shape[1], **kwargs),
        device=device,
        dtype=dtype,
        shape=shape,
    )


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(capsys, device, dtype, shape, affine):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_gradients(
        warp_module=nn.BatchNorm(shape[1], affine=affine),
        torch_module=_TORCH_MODULES[len(shape)](shape[1], affine=affine),
        device=device,
        dtype=dtype,
        shape=shape,
        weighted=True,
    )


@pytest.mark.parametrize("momentum", [0.1, 0.5])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_running_stats(capsys, device, shape, momentum):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module = nn.BatchNorm(shape[1], momentum=momentum).to(device)
    torch_module = _TORCH_MODULES[len(shape)](shape[1], momentum=momentum).to(device)
    # initial values
    utilities.check_arrays([torch_module.running_mean, torch_module.running_var], warp_module.buffers(), test="equal")
    # training mode: the running statistics are updated
    _train_both(warp_module, torch_module, shape=shape, device=device)
    utilities.check_arrays(
        [torch_module.running_mean, torch_module.running_var],
        [warp_module.running_mean.data, warp_module.running_var.data],
    )
    # evaluation mode: the running statistics are used, but not updated
    warp_module.eval()
    torch_module.eval()
    running_stats = [wp.clone(warp_module.running_mean.data), wp.clone(warp_module.running_var.data)]
    check_forward(warp_module=warp_module, torch_module=torch_module, device=device, dtype=wp.float32, shape=shape)
    check_gradients(
        warp_module=warp_module,
        torch_module=torch_module,
        device=device,
        dtype=wp.float32,
        shape=shape,
        weighted=True,
    )
    utilities.check_arrays(running_stats, [warp_module.running_mean.data, warp_module.running_var.data], test="equal")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_evaluation_without_running_stats(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    warp_module = nn.BatchNorm(5, track_running_stats=False).to(device).eval()
    torch_module = torch.nn.BatchNorm1d(5, track_running_stats=False).to(device).eval()
    assert warp_module.running_mean is None and warp_module.running_var is None
    assert not warp_module.buffers()
    # the batch statistics are used in evaluation mode (a single value per channel is allowed)
    check_forward(warp_module=warp_module, torch_module=torch_module, device=device, dtype=wp.float32, shape=(6, 5))
    warp_module(wp.zeros((1, 5), dtype=wp.float32, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffers(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # the buffers are moved to the device, and are part of the state dictionary
    module = nn.BatchNorm(3).to(device)
    state_dict = module.state_dict()
    assert list(state_dict.keys()) == ["weight", "bias", "running_mean", "running_var"]
    assert all(array.device == module.device for array in state_dict.values())
    assert state_dict["running_mean"] is module.running_mean.data
    # loading a state dictionary sets the buffers
    running_mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    module.load_state_dict({"running_mean": running_mean})
    np.testing.assert_array_equal(module.running_mean.data.numpy(), running_mean)
    # the loaded running mean is used for normalization in evaluation mode
    output = module.eval()(wp.array(np.tile(running_mean, (2, 1)), dtype=wp.float32, device=device))
    np.testing.assert_allclose(output.numpy(), np.zeros((2, 3)), atol=1e-6)


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requires_grad(capsys, device, requires_grad):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_requires_grad(
        warp_module=nn.BatchNorm(4, requires_grad=requires_grad),
        device=device,
        inputs=[wp.array(utilities.sample_array([2, 4, 3]), device=device, requires_grad=True)],
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_parameters(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    check_initialize_parameters(module_type=nn.BatchNorm, module_kwargs={"num_features": 4}, device=device)
    # the buffers are always initialized
    module = nn.BatchNorm(4, initialize_parameters=False).to(device)
    assert module.running_mean.data.numpy().tolist() == [0.0] * 4
    assert module.running_var.data.numpy().tolist() == [1.0] * 4
    # the default/initial values match PyTorch
    utilities.check_arrays(
        list(torch.nn.BatchNorm1d(4).parameters()), nn.BatchNorm(4).to(device).parameters(), test="equal"
    )


def test_invalid_input(capsys):
    batch_norm = nn.BatchNorm(4).to("cpu")
    assert batch_norm.training
    with pytest.raises(ValueError, match="shape"):
        batch_norm(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError, match="shape"):
        batch_norm(wp.zeros((4,), dtype=wp.float32, device="cpu"))
    # a single value per channel in training mode
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        batch_norm(wp.zeros((1, 4), dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        batch_norm(wp.zeros((1, 4, 1), dtype=wp.float32, device="cpu"))
    batch_norm.eval()(wp.zeros((1, 4), dtype=wp.float32, device="cpu"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_precision(capsys, device):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    # many values per channel, with a large mean compared to their deviation
    array = (np.random.rand(64, 2, 64, 64) + 10.0).astype(np.float32)
    warp_module = nn.BatchNorm(2).to(device)
    torch_module = torch.nn.BatchNorm2d(2, dtype=torch.float64)
    warp_output = warp_module(wp.array(array, device=device))
    torch_output = torch_module(torch.tensor(array, dtype=torch.float64))
    np.testing.assert_allclose(warp_output.numpy(), torch_output.detach().numpy(), atol=1e-4)
    np.testing.assert_allclose(warp_module.running_mean.data.numpy(), torch_module.running_mean.numpy(), rtol=1e-5)
    np.testing.assert_allclose(warp_module.running_var.data.numpy(), torch_module.running_var.numpy(), rtol=1e-4)
