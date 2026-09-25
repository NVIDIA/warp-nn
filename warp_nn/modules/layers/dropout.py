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

import numpy as np
import warp as wp

from warp_nn.modules.module import Module
from warp_nn.utils import contiguous


# The seed lives on the device and is incremented by a kernel (rather than passed as a launch argument),
# so that a captured graph draws a different mask on each replay.
# The mask is stored for the backward pass, since the seed changes after the forward pass.


@wp.kernel(enable_backward=False)
def _mask_kernel(seed: wp.array1d[wp.uint32], p: float, scale: float, mask: wp.array1d[Any]):
    i = wp.tid()
    state = wp.rand_init(wp.int32(seed[0]), i)
    if wp.randf(state) >= p:
        mask[i] = mask.dtype(scale)
    else:
        mask[i] = mask.dtype(0.0)


@wp.kernel(enable_backward=False)
def _increment_seed_kernel(seed: wp.array1d[wp.uint32]):
    seed[0] = seed[0] + wp.uint32(1)


@wp.kernel
def _apply_mask_kernel(input: wp.array1d[Any], mask: wp.array1d[Any], output: wp.array1d[Any]):
    i = wp.tid()
    output[i] = input[i] * mask[i]


_DTYPES = (wp.float16, wp.float32, wp.float64)
_MASK_KERNELS = {
    dtype: wp.overload(_mask_kernel, [wp.array1d(dtype=wp.uint32), float, float, wp.array1d(dtype=dtype)])
    for dtype in _DTYPES
}
_APPLY_MASK_KERNELS = {dtype: wp.overload(_apply_mask_kernel, [wp.array1d(dtype=dtype)] * 3) for dtype in _DTYPES}


class Dropout(Module):
    def __init__(self, p: float = 0.5, *, requires_grad: bool = True) -> None:
        r"""Randomly zero some of the input values during training.

        In training mode (see :py:meth:`~warp_nn.modules.module.Module.train`), each value is zeroed
        with probability :math:`p` (independently for each value and forward pass), and the kept values
        are scaled by :math:`\frac{1}{1 - p}`. In evaluation mode, the input array is returned as is.

        The random number generator is seeded from NumPy's global random number generator
        when the module is created.

        :param p: The probability of a value to be zeroed.
        :param requires_grad: Whether the cached output arrays of the module require gradients.

        :raises ValueError: If the probability is not in the interval [0, 1].
        """
        super().__init__(requires_grad=requires_grad)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"The dropout probability must be in the interval [0, 1], got {p}")
        self._p = p
        # runtime variables
        self._cache = {}
        self._seed = wp.array([np.random.randint(0, 2**32, dtype=np.uint32)], dtype=wp.uint32, device=self.device)

    @property
    def p(self) -> float:
        """The probability of a value to be zeroed."""
        return self._p

    def to(self, device: wp.Device) -> Dropout:
        """Move the module to the specified device.

        :param device: The device to move the module to.
        :return: The module itself.
        """
        super().to(device)
        self._seed = self._seed.to(self.device)
        return self

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array.

        :return: The output array, with same shape as the input array (the input array itself in evaluation mode).

        :raises TypeError: If the input array's data type is not supported.
        """
        if not self.training or self._p == 0.0:
            return input
        dtype = input.dtype
        if dtype not in _DTYPES:
            supported = ", ".join(t.__name__ for t in _DTYPES)
            raise TypeError(f"Unsupported data type {dtype.__name__} (supported: {supported})")
        shape = tuple(input.shape)
        key = (shape, dtype)
        # cache output and mask
        if key not in self._cache:
            output = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
            mask = wp.empty(input.size, dtype=dtype, device=self.device)
            self._cache[key] = (output, output.flatten(), mask)
        output, output_view, mask = self._cache[key]
        # launch kernels
        scale = 0.0 if self._p == 1.0 else 1.0 / (1.0 - self._p)
        wp.launch(
            _MASK_KERNELS[dtype],
            dim=mask.size,
            inputs=[self._seed, self._p, scale],
            outputs=[mask],
            device=self.device,
            record_tape=False,
        )
        wp.launch(_increment_seed_kernel, dim=1, inputs=[self._seed], device=self.device, record_tape=False)
        wp.launch(
            _APPLY_MASK_KERNELS[dtype],
            dim=mask.size,
            inputs=[contiguous(input).flatten(), mask],
            outputs=[output_view],
            device=self.device,
        )
        return output
