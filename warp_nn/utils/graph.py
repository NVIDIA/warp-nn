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

import warp as wp

from warp_nn.utils.device import parse_device


class ScopedCapture:
    def __init__(
        self,
        *,
        device: str | wp.Device | None = None,
        stream: wp.Stream | None = None,
        force_module_load: bool | None = None,
        external: bool = False,
        capture_mode: wp.CaptureMode = wp.CaptureMode.THREAD_LOCAL,
        enabled: bool = True,
    ) -> None:
        """Context manager for capturing CUDA graphs.

        Adapted from the Warp's :py:class:`~warp.ScopedCapture` implementation
        to support enabling/disabling the capture.

        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param stream: CUDA stream to capture on. If not specified, the default stream will be used.
        :param force_module_load: Whether to force loading of all kernels before capture.
        :param external: Whether the capture was already started externally.
        :param capture_mode: CUDA stream capture mode. Use :py:attr:`warp.CaptureMode.RELAXED` when composing
            with libraries that perform capture-unsafe runtime calls during the capture.
        :param enabled: Whether to enable the capture. If disabled, the capture will be skipped and the graph will be None.
        """
        self._graph = None
        self._enabled = enabled
        if enabled:
            self._device = parse_device(device)
            self._stream = stream
            self._force_module_load = force_module_load
            self._external = external
            self._capture_mode = capture_mode
            self._active = False

    @property
    def graph(self) -> wp._src.context.Graph | None:
        """Captured graph"""
        return self._graph

    def __enter__(self) -> "ScopedCapture":
        """Begin capture of a CUDA graph"""
        if self._enabled:
            self._graph = None
            wp.capture_begin(
                device=self._device,
                stream=self._stream,
                force_module_load=self._force_module_load,
                external=self._external,
                capture_mode=self._capture_mode,
            )
            self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """End capture of a CUDA graph"""
        if self._enabled:
            try:
                self._graph = wp.capture_end(device=self._device, stream=self._stream)
            finally:
                self._active = False
