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

from __future__ import annotations

from typing import Any

import warp as wp

from warp_nn.optimizers.optimizer import Optimizer
from warp_nn.utils import KernelConfig, ScopedCapture, resolve_dim


def _create_kernels(config: KernelConfig, *, momentum: float, dampening: float, weight_decay: float):
    @wp.kernel(enable_backward=False)
    def increase_timestep(t: wp.array1d[Any]):
        t[0] += 1.0

    @wp.kernel(enable_backward=False)
    def optimizer_step(
        parameters: wp.array1d[Any],
        gradients: wp.array1d[Any],
        velocity: wp.array1d[Any],
        timestep: wp.array1d[Any],
        lr: wp.array1d[Any],
    ):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tiled_parameters = wp.tile_load(parameters, shape=shape, offset=offset)
        tiled_gradients = wp.tile_load(gradients, shape=shape, offset=offset)
        tiled_velocity = wp.tile_load(velocity, shape=shape, offset=offset)
        if wp.static(weight_decay):
            tiled_gradients = tiled_gradients + wp.static(weight_decay) * tiled_parameters
        if wp.static(momentum):
            if timestep[0] > 1.0:
                tiled_velocity = wp.static(momentum) * tiled_velocity + wp.static(1.0 - dampening) * tiled_gradients
            else:
                tiled_velocity = tiled_gradients
            wp.tile_store(velocity, tiled_velocity, offset=offset)
            tiled_gradients = tiled_velocity
        wp.tile_store(parameters, tiled_parameters - lr[0] * tiled_gradients, offset=offset)

    return increase_timestep, optimizer_step


class SGD(Optimizer):
    def __init__(
        self,
        parameters: list[wp.array],
        *,
        lr: float = 1e-3,
        device: str | wp.Device | None = None,
        max_norm: float | None = None,
        disable_graph: bool = False,
        # optimizer-specific parameters
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
    ) -> None:
        """SGD optimizer.

        :param parameters: Model parameters.
        :param lr: Learning rate.
        :param device: Device to use for the optimizer.
        :param disable_graph: Whether to disable graph capture.
        :param momentum: Momentum factor.
        :param dampening: Dampening factor.
        :param weight_decay: Weight decay factor.
        """
        super().__init__(parameters, lr=lr, device=device, max_norm=max_norm, disable_graph=disable_graph)

        self._momentum = momentum
        self._dampening = dampening
        self._weight_decay = weight_decay
        self._velocity = [wp.zeros_like(param, requires_grad=False) for param in self._parameters]
        self._timestep = wp.zeros((1,), dtype=wp.float32, device=self.device)

        # runtime variables
        self._kernel_increase_timestep, self._kernel_step = _create_kernels(
            self._config, momentum=self._momentum, dampening=self._dampening, weight_decay=self._weight_decay
        )

    def step(self, *, lr: float | None = None) -> None:
        """Perform an optimization step to update parameters.

        :param lr: Learning rate.
        """
        # update learning rate
        if lr is not None:
            self._lr.fill_(lr)
        # perform optimization step
        if self._graph_step is None:
            with ScopedCapture(device=self.device, enabled=self._device.is_cuda and not self._disable_graph) as capture:
                if self._max_norm is not None:
                    self._launch_clip_by_total_norm()
                wp.launch(
                    self._kernel_increase_timestep,
                    dim=1,
                    inputs=[self._timestep],
                    device=self._device,
                    block_dim=self._config.block_dim,
                )
                for parameters, gradients, velocity in zip(self._parameters, self._gradients, self._velocity):
                    wp.launch_tiled(
                        self._kernel_step,
                        dim=resolve_dim(config=self._config, shape=parameters.shape, tiled=True),
                        inputs=[parameters, gradients, velocity, self._timestep, self._lr],
                        device=self._device,
                        block_dim=self._config.block_dim,
                    )
            self._graph_step = capture.graph
        if self._graph_step is not None:
            wp.capture_launch(self._graph_step)

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError
