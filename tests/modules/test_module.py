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

import hypothesis
import hypothesis.strategies as st
import pytest

import numpy as np
import warp as wp

import warp_nn.nn as nn

from .. import utilities


class ModuleA(nn.Module):  # parameters: 0, buffers: 0, modules: 0
    def __init__(self):
        super().__init__()
        super().__post_init__()


class ModuleB(nn.Module):  # parameters: 1, buffers: 0, modules: 0
    def __init__(self):
        super().__init__()
        self.param_b0 = nn.Parameter(wp.full(shape=(1, 1), value=0.0, dtype=wp.float32))
        super().__post_init__()


class ModuleC(nn.Module):  # parameters: 0, buffers: 1, modules: 0
    def __init__(self):
        super().__init__()
        self.buffer_c0 = nn.Buffer(wp.full(shape=(3,), value=3.0, dtype=wp.float32))
        super().__post_init__()


class ModuleD(nn.Module):  # parameters: 3 (2 + (1 + 0)), buffers: 1 (0 + (0 + 1)), modules: 2
    def __init__(self):
        super().__init__()
        self.module_b = ModuleB()
        self.module_c = ModuleC()
        self.param_d0 = nn.Parameter(wp.full(shape=(2, 2), value=1.0, dtype=wp.float32))
        self.param_d1 = nn.Parameter(wp.full(shape=(3, 3), value=2.0, dtype=wp.float32))
        super().__post_init__()


class ModuleE(nn.Module):  # parameters: 5 (1 + (0 + 1 + 3)), buffers: 3 (2 + (0 + 0 + 1)), modules: 3
    def __init__(self):
        super().__init__()
        self.module_a = ModuleA()
        self.module_b = ModuleB()
        self.module_d = ModuleD()
        self.param_e0 = nn.Parameter(wp.full(shape=(4, 4), value=4.0, dtype=wp.float32))
        self.buffer_e0 = nn.Buffer(wp.full(shape=(2, 2), value=6, dtype=wp.int32))
        self.buffer_e1 = nn.Buffer(wp.full(shape=(2,), value=5.0, dtype=wp.float32))
        super().__post_init__()


@pytest.fixture
def modules():
    return [ModuleA(), ModuleB(), ModuleC(), ModuleD(), ModuleE()]


@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize("include_submodules", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parameters(capsys, device, include_submodules, as_array, modules: list[nn.Module]):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    def _check(parameters):
        assert all(
            isinstance(parameter, wp.array) if as_array else isinstance(parameter, nn.Parameter)
            for parameter in parameters
        )
        assert all(parameter.device.is_cuda == (device == "cuda") for parameter in parameters)

    module_a, module_b, module_c, module_d, module_e = [module.to(device) for module in modules]
    # number of parameters
    assert len(module_a.parameters(include_submodules=include_submodules, as_array=as_array)) == 0
    assert len(module_b.parameters(include_submodules=include_submodules, as_array=as_array)) == 1
    assert len(module_c.parameters(include_submodules=include_submodules, as_array=as_array)) == 0
    assert len(module_d.parameters(include_submodules=include_submodules, as_array=as_array)) == (
        3 if include_submodules else 2
    )
    assert len(module_e.parameters(include_submodules=include_submodules, as_array=as_array)) == (
        5 if include_submodules else 1
    )
    # class/device
    _check(module_a.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_b.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_c.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_d.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_e.parameters(include_submodules=include_submodules, as_array=as_array))
    # moving the module to another device keeps the parameter instances, but replaces their data
    if utilities.is_device_available("cuda"):
        parameters = module_e.parameters(as_array=False)
        arrays = [parameter.data for parameter in parameters]
        module_e.to("cpu" if device == "cuda" else "cuda")
        assert all(a is b for a, b in zip(module_e.parameters(as_array=False), parameters))
        for parameter, array in zip(parameters, arrays):
            assert parameter.device == module_e.device
            assert parameter.data is not array
            assert parameter.dtype == array.dtype
            assert parameter.requires_grad == array.requires_grad
            assert parameter.data.numpy().tolist() == array.numpy().tolist()


@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize("include_submodules", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffers(capsys, device, include_submodules, as_array, modules: list[nn.Module]):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    def _check(buffers):
        assert all(isinstance(buffer, wp.array) if as_array else isinstance(buffer, nn.Buffer) for buffer in buffers)
        assert all(buffer.device.is_cuda == (device == "cuda") for buffer in buffers)

    module_a, module_b, module_c, module_d, module_e = [module.to(device) for module in modules]
    # number of buffers
    assert len(module_a.buffers(include_submodules=include_submodules, as_array=as_array)) == 0
    assert len(module_b.buffers(include_submodules=include_submodules, as_array=as_array)) == 0
    assert len(module_c.buffers(include_submodules=include_submodules, as_array=as_array)) == 1
    assert len(module_d.buffers(include_submodules=include_submodules, as_array=as_array)) == (
        1 if include_submodules else 0
    )
    assert len(module_e.buffers(include_submodules=include_submodules, as_array=as_array)) == (
        3 if include_submodules else 2
    )
    # class/device
    _check(module_a.buffers(include_submodules=include_submodules, as_array=as_array))
    _check(module_b.buffers(include_submodules=include_submodules, as_array=as_array))
    _check(module_c.buffers(include_submodules=include_submodules, as_array=as_array))
    _check(module_d.buffers(include_submodules=include_submodules, as_array=as_array))
    _check(module_e.buffers(include_submodules=include_submodules, as_array=as_array))
    # moving the module to another device keeps the buffer instances, but replaces their data
    if utilities.is_device_available("cuda"):
        buffers = module_e.buffers(as_array=False)
        arrays = [buffer.data for buffer in buffers]
        module_e.to("cpu" if device == "cuda" else "cuda")
        assert all(a is b for a, b in zip(module_e.buffers(as_array=False), buffers))
        for buffer, array in zip(buffers, arrays):
            assert buffer.device == module_e.device
            assert buffer.data is not array
            assert buffer.dtype == array.dtype
            assert buffer.requires_grad == array.requires_grad
            assert buffer.data.numpy().tolist() == array.numpy().tolist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_state_dict(capsys, device, modules: list[nn.Module]):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module_a, module_b, module_c, module_d, module_e = [module.to(device) for module in modules]
    # state dict
    assert set(module_a.state_dict().keys()) == set()
    assert set(module_b.state_dict().keys()) == set(["param_b0"])
    assert set(module_c.state_dict().keys()) == set(["buffer_c0"])
    assert set(module_d.state_dict().keys()) == set(["module_b.param_b0", "module_c.buffer_c0", "param_d0", "param_d1"])
    assert set(module_e.state_dict().keys()) == set(
        [
            "module_b.param_b0",
            "module_d.module_b.param_b0",
            "module_d.module_c.buffer_c0",
            "module_d.param_d0",
            "module_d.param_d1",
            "param_e0",
            "buffer_e0",
            "buffer_e1",
        ]
    )


def test_train_eval(capsys):
    module = nn.Sequential(
        nn.Linear(4, 3),
        nn.Sequential(nn.Dropout(0.5), nn.BatchNorm(3)),
        nn.ReLU(),
    )
    modules = [module, *module.modules(), *list(module.modules())[1].modules()]
    # training mode by default
    assert all(m.training for m in modules)
    # the mode is set recursively, and the module itself is returned
    assert module.eval() is module
    assert not any(m.training for m in modules)
    assert module.train() is module
    assert all(m.training for m in modules)
    assert module.train(False) is module
    assert not any(m.training for m in modules)


@pytest.mark.parametrize(
    "module_factory, shape",
    [
        (lambda: nn.Linear(4, 3), (2, 4)),
        (lambda: nn.ReLU(), (2, 4)),
        (lambda: nn.BatchNorm(4), (2, 4, 3)),
        (lambda: nn.Dropout(0.5), (2, 4)),
        (lambda: nn.MaxPool1D(2), (2, 4, 6)),
        (lambda: nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3)), (2, 4)),
    ],
    ids=["Linear", "ReLU", "BatchNorm", "Dropout", "MaxPool1D", "Sequential"],
)
def test_move_after_forward(capsys, module_factory, shape):
    if not utilities.is_device_available("cuda"):
        pytest.skip("Device 'cuda' is not available")
    # the arrays cached by a forward pass must not be reused after moving the module to another device
    module = module_factory()
    for device in ["cpu", "cuda", "cpu"]:
        output = module.to(device)(wp.ones(shape, dtype=wp.float32, device=device))
        assert output.device == wp.get_device(device)
        assert all(array.device == wp.get_device(device) for array in module.parameters() + module.buffers())
