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

from abc import ABC, abstractmethod

import warp as wp

from warp_nn.utils import KernelConfig, ScopedCapture, get_kernel_config, parse_device, resolve_dim


def _create_clip_by_total_norm_kernels(config: KernelConfig, *, max_norm: float):
    @wp.func
    def clip_by_norm(x: wp.float32, sum_squares: wp.float32) -> wp.float32:
        norm = wp.sqrt(sum_squares)
        if norm > wp.static(max_norm):
            return x / norm * wp.static(max_norm)
        return x

    @wp.kernel(enable_backward=False)
    def sum_squares(gradients: wp.array1d[Any], sum_squares: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tiled_gradients = wp.tile_load(gradients, shape=shape, offset=offset)
        wp.tile_atomic_add(sum_squares, wp.tile_sum(wp.tile_map(wp.mul, tiled_gradients, tiled_gradients)))

    @wp.kernel(enable_backward=False)
    def clip_by_total_norm(gradients: wp.array1d[Any], sum_squares: wp.array1d[Any]):
        i = wp.tid()
        shape = (wp.static(config.tile_1d[0]),)
        offset = (i * wp.static(config.tile_1d[0]),)
        tiled_sum_squares = wp.tile_broadcast(wp.tile_load(sum_squares, shape=(1,)), shape=shape)
        tiled_gradients = wp.tile_load(gradients, shape=shape, offset=offset)
        tiled_gradients = wp.tile_map(wp.static(clip_by_norm), tiled_gradients, tiled_sum_squares)
        wp.tile_store(gradients, tiled_gradients, offset=offset)

    return sum_squares, clip_by_total_norm


class Optimizer(ABC):
    def __init__(
        self,
        parameters: list[wp.array],
        *,
        lr: float = 1e-3,
        device: str | wp.Device | None = None,
        max_norm: float | None = None,
        disable_graph: bool = False,
    ) -> None:
        """Base class for all optimizers.

        :param parameters: Model parameters.
        :param lr: Learning rate.
        :param device: Device to use for the optimizer.
        :param disable_graph: Whether to disable graph capture.
        """
        self._device = parse_device(device)
        self._parameters = [param.flatten() for param in parameters]
        self._gradients = [param.grad.flatten() for param in self._parameters]
        self._lr = wp.array([lr], dtype=wp.float32, device=self._device)
        self._disable_graph = disable_graph
        self._max_norm = max_norm

        # runtime variables
        self._config = get_kernel_config()
        self._graph_step = None
        if self._max_norm is not None:
            self._configure_clip_by_total_norm(self._max_norm)

    @property
    def device(self) -> wp.Device:
        return self._device

    @property
    def parameters(self) -> list[wp.array]:
        return self._parameters

    @property
    def gradients(self) -> list[wp.array]:
        return self._gradients

    @abstractmethod
    def step(self, *, lr: float | None = None) -> None:
        """Perform an optimization step to update parameters.

        :param lr: Learning rate.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def _configure_clip_by_total_norm(self, max_norm: float):
        self._max_norm = max_norm
        self._graph_step = None  # captured (if any) with the previous kernels and arrays
        self._graph_clip_by_total_norm = None
        self._array_sum_squares = wp.zeros((1,), dtype=wp.float32, device=self._device)
        self._kernel_sum_squares, self._kernel_clip_by_total_norm = _create_clip_by_total_norm_kernels(
            self._config, max_norm=self._max_norm
        )

    def _launch_clip_by_total_norm(self) -> None:
        """Launch the gradient clipping kernels (without graph capture/replay) for the current `max_norm`."""
        self._array_sum_squares.zero_()
        for gradient in self._gradients:
            wp.launch_tiled(
                self._kernel_sum_squares,
                dim=resolve_dim(config=self._config, shape=gradient.shape, tiled=True),
                inputs=[gradient],
                outputs=[self._array_sum_squares],
                device=self._device,
                block_dim=self._config.block_dim,
            )
        for gradient in self._gradients:
            wp.launch_tiled(
                self._kernel_clip_by_total_norm,
                dim=resolve_dim(config=self._config, shape=gradient.shape, tiled=True),
                inputs=[gradient, self._array_sum_squares],
                device=self._device,
                block_dim=self._config.block_dim,
            )

    def clip_by_total_norm(self, max_norm: float, *, disable_graph: bool = False):
        """Clip (scaling down) parameters' gradients in-place by their total norm.

        https://arxiv.org/abs/1211.5063

        .. note::

            The given ``max_norm`` also becomes the one used by subsequent :py:meth:`step` calls.
            Changing it creates new kernels (compiled on first use) and discards the captured graphs.

        :param max_norm: Maximum global norm.
        :param disable_graph: Whether to disable graph capture (e.g.: to capture it as part of an external graph).
        """
        # create kernels if not already done or if `max_norm` has changed
        if max_norm != self._max_norm:
            self._configure_clip_by_total_norm(max_norm)
        # clip gradients
        if self._device.is_cuda and not disable_graph:
            if self._graph_clip_by_total_norm is None:
                with ScopedCapture(device=self._device) as capture:
                    self._launch_clip_by_total_norm()
                self._graph_clip_by_total_norm = capture.graph
            wp.capture_launch(self._graph_clip_by_total_norm)
        else:
            self._launch_clip_by_total_norm()
