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

import torch

import warp as wp

import warp_nn.nn as nn
from warp_nn.optimizers import SGD

from .. import utilities


@hypothesis.given(learning_rate=st.floats(min_value=1e-5, max_value=1e-1))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
# optimizer-specific parameters
@pytest.mark.parametrize("max_norm", [None, 1.0])
@pytest.mark.parametrize("weight_decay", [0.0, 0.9])
@pytest.mark.parametrize("dampening", [0.0, 0.8])
@pytest.mark.parametrize("momentum", [0.0, 0.7])
# test-specific parameters
@pytest.mark.parametrize("disable_graph", [True, False])
@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("dtype", [wp.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_step(capsys, device, dtype, ndim, disable_graph, momentum, dampening, weight_decay, max_norm, learning_rate):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    # loss function
    @wp.kernel
    def _sum_loss(a: wp.array2d(dtype=float), loss: wp.array1d(dtype=float)):
        i, j = wp.tid()
        wp.atomic_add(loss, 0, a[i, j])

    # static parameters
    in_features = 2
    out_features = 2
    batch_size = 10
    # create modules
    torch_module = torch.nn.Linear(in_features=in_features, out_features=out_features)
    torch_module.to(device)
    warp_module = nn.Linear(in_features=in_features, out_features=out_features)
    warp_module.to(device)
    # init parameters to same values
    utilities.init_parameters(torch_module.parameters())
    utilities.init_parameters(warp_module.parameters(as_array=True))
    torch_old_parameters = [torch.clone(parameter) for parameter in torch_module.parameters()]
    warp_old_parameters = [wp.clone(parameter) for parameter in warp_module.parameters(as_array=True)]
    # create optimizers
    torch_optimizer = torch.optim.SGD(
        torch_module.parameters(),
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
    )
    warp_optimizer = SGD(
        warp_module.parameters(as_array=True),
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        max_norm=max_norm,
        device=device,
        disable_graph=disable_graph,
    )
    warp_loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True, device=device)
    for i in range(10):
        msg = f"optimization step {i+1}/10"
        # create inputs
        array_input = utilities.sample_array(shape=[batch_size, in_features], dtype=dtype)
        torch_input = torch.tensor(array_input, device=device, requires_grad=True)
        warp_input = wp.array(array_input, device=device, requires_grad=True)
        # compute loss
        # - torch
        torch_output = torch_module(torch_input)
        torch_loss = torch_output.sum()
        torch_loss.backward()
        # - warp
        warp_loss.zero_()
        with wp.Tape() as warp_tape:
            warp_output = warp_module(warp_input)
            wp.launch(_sum_loss, dim=warp_output.shape, inputs=[warp_output, warp_loss], device=device)
        warp_tape.backward(warp_loss)
        # step optimizers
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(torch_module.parameters(), max_norm)
        torch_optimizer.step()
        warp_optimizer.step()
        # check gradients
        utilities.check_arrays(
            [parameter.grad for parameter in torch_module.parameters()],
            [parameter.grad for parameter in warp_module.parameters(as_array=True)],
            flatten=True,
            msg=msg,
        )
        # - reset gradients
        torch_optimizer.zero_grad()
        warp_tape.zero()
        # check parameters
        torch_new_parameters = [torch.clone(parameter) for parameter in torch_module.parameters()]
        warp_new_parameters = [wp.clone(parameter) for parameter in warp_module.parameters(as_array=True)]
        # - check different between both frameworks
        utilities.check_arrays(torch_new_parameters, warp_new_parameters, flatten=True, msg=msg)
        # - check different between old and new parameters
        utilities.check_arrays(torch_old_parameters, torch_new_parameters, test="not-equal", msg=msg)
        utilities.check_arrays(warp_old_parameters, warp_new_parameters, test="not-equal", msg=msg)
        torch_old_parameters = torch_new_parameters
        warp_old_parameters = warp_new_parameters
