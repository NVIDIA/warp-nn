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
    def activation(x: float):
        return wp.tanh(x)

    @wp.kernel
    def kernel(
        input: wp.array2d[float],
        hidden: wp.array2d[float],
        weight_ih: wp.array2d[float],
        weight_hh: wp.array2d[float],
        bias_ih: wp.array2d[float],
        bias_hh: wp.array2d[float],
        output: wp.array2d[float],
    ):
        i, j = wp.tid()
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        gate_ih = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_ih, input, index=(i, j))
        gate_hh = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hh, hidden, index=(i, j))
        if wp.static(include_bias):
            shape_T = (wp.static(config.tile_2d[1]), wp.static(config.tile_2d[0]))
            shape_b = (wp.static(config.tile_2d[1]), 1)
            offset_b = (j * wp.static(config.tile_2d[1]), 0)
            gate_ih += wp.tile_broadcast(wp.tile_load(bias_ih, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hh += wp.tile_broadcast(wp.tile_load(bias_hh, shape=shape_b, offset=offset_b), shape=shape_T)
        wp.tile_store(output, wp.tile_transpose(wp.tile_map(activation, gate_ih + gate_hh)), offset=offset)

    return kernel


class RNNCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a Elman's Recurrent Neural Network (RNN) cell.

        .. math::

            \text{RNNCell}(x, h) = \text{tanh}(W_{ih} \, x + b_{ih} + W_{hh} \, h + b_{hh})

        |hr|

        Learnable parameters:

        .. list-table::
            :header-rows: 1

            * -
              - Name
              - Shape
              - Description
            * - :math:`W_{ih}`
              - ``weight_ih``
              - ``(hidden_size, input_size)``
              - Input-to-hidden weights
            * - :math:`W_{hh}`
              - ``weight_hh``
              - ``(hidden_size, hidden_size)``
              - Hidden-to-hidden weights
            * - :math:`b_{ih}`
              - ``bias_ih``
              - ``(hidden_size, 1)``
              - Input-to-hidden bias. Only if ``bias`` is true
            * - :math:`b_{hh}`
              - ``bias_hh``
              - ``(hidden_size, 1)``
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
            wp.empty(shape=(self.hidden_size, self.input_size), dtype=wp.float32, device=self.device),
            requires_grad=self.requires_grad,
        )
        self.weight_hh = Parameter(
            wp.empty(shape=(self.hidden_size, self.hidden_size), dtype=wp.float32, device=self.device),
            requires_grad=self.requires_grad,
        )
        self.register_parameter(name="weight_ih", parameter=self.weight_ih)
        self.register_parameter(name="weight_hh", parameter=self.weight_hh)
        # - biases
        if bias:
            self.bias_ih = Parameter(
                wp.empty(shape=(self.hidden_size, 1), dtype=wp.float32, device=self.device),
                requires_grad=self.requires_grad,
            )
            self.bias_hh = Parameter(
                wp.empty(shape=(self.hidden_size, 1), dtype=wp.float32, device=self.device),
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
                self.weight_ih.data,
                self.weight_hh.data,
                self.bias_ih.data if self.bias_ih else None,
                self.bias_hh.data if self.bias_hh else None,
            ],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
