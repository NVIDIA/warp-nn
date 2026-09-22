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

import warp as wp

from warp_nn.initializers import kaiming_uniform
from warp_nn.modules.module import Module
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import KernelConfig, get_kernel_config, resolve_dim

from ._common import tile_transposed_gemm_2d


def _create_kernels(config: KernelConfig, *, include_bias: bool):

    @wp.func
    def sigmoid(x: float):
        return 1.0 / (1.0 + wp.exp(-x))

    @wp.func
    def compute_hidden_state(
        hidden: float, gate_ir: float, gate_hr: float, gate_iz: float, gate_hz: float, gate_in: float, gate_hn: float
    ):
        r = sigmoid(gate_ir + gate_hr)
        z = sigmoid(gate_iz + gate_hz)
        n = wp.tanh(gate_in + r * gate_hn)
        return (1.0 - z) * n + z * hidden

    @wp.kernel
    def kernel(
        input: wp.array2d[float],
        hidden: wp.array2d[float],
        weight_ir: wp.array2d[float],
        weight_iz: wp.array2d[float],
        weight_in: wp.array2d[float],
        weight_hr: wp.array2d[float],
        weight_hz: wp.array2d[float],
        weight_hn: wp.array2d[float],
        bias_ir: wp.array2d[float],
        bias_iz: wp.array2d[float],
        bias_in: wp.array2d[float],
        bias_hr: wp.array2d[float],
        bias_hz: wp.array2d[float],
        bias_hn: wp.array2d[float],
        output: wp.array2d[float],
    ):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        t_hidden = wp.tile_transpose(wp.tile_load(hidden, shape=shape, offset=offset))
        gate_ir = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_ir, input, index=(i, j))
        gate_iz = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_iz, input, index=(i, j))
        gate_in = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_in, input, index=(i, j))
        gate_hr = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hr, hidden, index=(i, j))
        gate_hz = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hz, hidden, index=(i, j))
        gate_hn = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hn, hidden, index=(i, j))
        if wp.static(include_bias):
            shape_T = (wp.static(config.tile_2d[1]), wp.static(config.tile_2d[0]))
            shape_b = (wp.static(config.tile_2d[1]), 1)
            offset_b = (j * wp.static(config.tile_2d[1]), 0)
            gate_ir += wp.tile_broadcast(wp.tile_load(bias_ir, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_iz += wp.tile_broadcast(wp.tile_load(bias_iz, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_in += wp.tile_broadcast(wp.tile_load(bias_in, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hr += wp.tile_broadcast(wp.tile_load(bias_hr, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hz += wp.tile_broadcast(wp.tile_load(bias_hz, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hn += wp.tile_broadcast(wp.tile_load(bias_hn, shape=shape_b, offset=offset_b), shape=shape_T)
        h = wp.tile_map(compute_hidden_state, t_hidden, gate_ir, gate_hr, gate_iz, gate_hz, gate_in, gate_hn)
        wp.tile_store(output, wp.tile_transpose(h), offset=offset)

    return kernel


class GRUCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a Gated Recurrent Unit (GRU) cell.

        .. math::

            \text{GRUCell}(x, h) = h'

        where

        .. math::

            \begin{array}{ll}
                r = \sigma(W_{ir} \, x + b_{ir} + W_{hr} \, h + b_{hr}) \\
                z = \sigma(W_{iz} \, x + b_{iz} + W_{hz} \, h + b_{hz}) \\
                n = \tanh(W_{in} \, x + b_{in} + r \odot (W_{hn} \, h + b_{hn})) \\
                h' = (1 - z) \odot n + z \odot h
            \end{array}

        and :math:`\sigma` is the sigmoid function and :math:`\odot` is the element-wise product.

        |hr|

        Learnable parameters:

        .. list-table::
            :header-rows: 1

            * -
              - Name
              - Shape
              - Description
            * - :math:`W_{ir}, W_{iz}, W_{in}`
              - ``weight_ih``
              - ``(3 * hidden_size, input_size)``
              - Input-to-hidden weights
            * - :math:`W_{hr}, W_{hz}, W_{hn}`
              - ``weight_hh``
              - ``(3 * hidden_size, hidden_size)``
              - Hidden-to-hidden weights
            * - :math:`b_{ir}, b_{iz}, b_{in}`
              - ``bias_ih``
              - ``(3 * hidden_size, 1)``
              - Input-to-hidden bias. Only if ``bias`` is true
            * - :math:`b_{hr}, b_{hz}, b_{hn}`
              - ``bias_hh``
              - ``(3 * hidden_size, 1)``
              - Hidden-to-hidden bias. Only if ``bias`` is true

        The parameters are initialized from the uniform distribution :math:`u(-k, k)`
        where :math:`k = \frac{1}{\sqrt{\text{hidden\_size}}}`.

        |hr|

        :param input_size: The number of input features.
        :param hidden_size: The number of hidden features.
        :param bias: Whether to include a bias term.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self.input_size = input_size
        self.hidden_size = hidden_size
        # create/register parameters
        # - weights
        self.weight_ih = Parameter(
            wp.empty(shape=(3 * self.hidden_size, self.input_size), dtype=wp.float32, device=self.device),
            requires_grad=self.requires_grad,
        )
        self.weight_hh = Parameter(
            wp.empty(shape=(3 * self.hidden_size, self.hidden_size), dtype=wp.float32, device=self.device),
            requires_grad=self.requires_grad,
        )
        self.register_parameter(name="weight_ih", parameter=self.weight_ih)
        self.register_parameter(name="weight_hh", parameter=self.weight_hh)
        # - biases
        if bias:
            self.bias_ih = Parameter(
                wp.empty(shape=(3 * self.hidden_size, 1), dtype=wp.float32, device=self.device),
                requires_grad=self.requires_grad,
            )
            self.bias_hh = Parameter(
                wp.empty(shape=(3 * self.hidden_size, 1), dtype=wp.float32, device=self.device),
                requires_grad=self.requires_grad,
            )
            self.register_parameter(name="bias_ih", parameter=self.bias_ih)
            self.register_parameter(name="bias_hh", parameter=self.bias_hh)
        else:
            self.bias_ih = None
            self.bias_hh = None
        # set default/initial values
        if initialize_parameters:
            self._initialize_parameters()
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernel = _create_kernels(self._config, include_bias=self.bias_ih is not None)
        self._slices = (
            slice(0 * self.hidden_size, 1 * self.hidden_size),
            slice(1 * self.hidden_size, 2 * self.hidden_size),
            slice(2 * self.hidden_size, 3 * self.hidden_size),
        )

    def _initialize_parameters(self):
        kaiming_uniform(self.weight_ih.data, mode="scale", scale=1.0 / self.hidden_size)
        kaiming_uniform(self.weight_hh.data, mode="scale", scale=1.0 / self.hidden_size)
        if self.bias_ih:
            kaiming_uniform(self.bias_ih.data, mode="scale", scale=1.0 / self.hidden_size)
            kaiming_uniform(self.bias_hh.data, mode="scale", scale=1.0 / self.hidden_size)

    def __call__(self, input: wp.array, hidden: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, input_size)``.
        :param hidden: The initial hidden state array, with shape ``(batch_size, hidden_size)``.

        :return: The next hidden state array, with shape ``(batch_size, hidden_size)``.
        """
        dtype = input.dtype
        shape = (input.shape[0], self.hidden_size)
        key = (shape, dtype)
        # cache output
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
        output = self._cache[key]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[
                input,
                hidden,
                self.weight_ih.data[self._slices[0]],
                self.weight_ih.data[self._slices[1]],
                self.weight_ih.data[self._slices[2]],
                self.weight_hh.data[self._slices[0]],
                self.weight_hh.data[self._slices[1]],
                self.weight_hh.data[self._slices[2]],
                self.bias_ih.data[self._slices[0]] if self.bias_ih else None,
                self.bias_ih.data[self._slices[1]] if self.bias_ih else None,
                self.bias_ih.data[self._slices[2]] if self.bias_ih else None,
                self.bias_hh.data[self._slices[0]] if self.bias_hh else None,
                self.bias_hh.data[self._slices[1]] if self.bias_hh else None,
                self.bias_hh.data[self._slices[2]] if self.bias_hh else None,
            ],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
