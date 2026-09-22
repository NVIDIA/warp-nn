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
    def compute_cell_state(
        cell: float, gate_ii: float, gate_hi: float, gate_if: float, gate_hf: float, gate_ig: float, gate_hg: float
    ):
        i = sigmoid(gate_ii + gate_hi)
        f = sigmoid(gate_if + gate_hf)
        g = wp.tanh(gate_ig + gate_hg)
        return f * cell + i * g

    @wp.func
    def compute_hidden_state(cell: float, gate_io: float, gate_ho: float):
        o = sigmoid(gate_io + gate_ho)
        return o * wp.tanh(cell)

    @wp.kernel
    def kernel(
        input: wp.array2d[float],
        hidden: wp.array2d[float],
        cell: wp.array2d[float],
        weight_ii: wp.array2d[float],
        weight_if: wp.array2d[float],
        weight_ig: wp.array2d[float],
        weight_io: wp.array2d[float],
        weight_hi: wp.array2d[float],
        weight_hf: wp.array2d[float],
        weight_hg: wp.array2d[float],
        weight_ho: wp.array2d[float],
        bias_ii: wp.array2d[float],
        bias_if: wp.array2d[float],
        bias_ig: wp.array2d[float],
        bias_io: wp.array2d[float],
        bias_hi: wp.array2d[float],
        bias_hf: wp.array2d[float],
        bias_hg: wp.array2d[float],
        bias_ho: wp.array2d[float],
        output_hidden: wp.array2d[float],
        output_cell: wp.array2d[float],
    ):
        i, j = wp.tid()
        shape = (wp.static(config.tile_2d[0]), wp.static(config.tile_2d[1]))
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        t_cell = wp.tile_transpose(wp.tile_load(cell, shape=shape, offset=offset))
        gate_ii = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_ii, input, index=(i, j))
        gate_if = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_if, input, index=(i, j))
        gate_ig = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_ig, input, index=(i, j))
        gate_io = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_io, input, index=(i, j))
        gate_hi = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hi, hidden, index=(i, j))
        gate_hf = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hf, hidden, index=(i, j))
        gate_hg = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_hg, hidden, index=(i, j))
        gate_ho = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight_ho, hidden, index=(i, j))
        if wp.static(include_bias):
            shape_T = (wp.static(config.tile_2d[1]), wp.static(config.tile_2d[0]))
            shape_b = (wp.static(config.tile_2d[1]), 1)
            offset_b = (j * wp.static(config.tile_2d[1]), 0)
            gate_ii += wp.tile_broadcast(wp.tile_load(bias_ii, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_if += wp.tile_broadcast(wp.tile_load(bias_if, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_ig += wp.tile_broadcast(wp.tile_load(bias_ig, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_io += wp.tile_broadcast(wp.tile_load(bias_io, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hi += wp.tile_broadcast(wp.tile_load(bias_hi, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hf += wp.tile_broadcast(wp.tile_load(bias_hf, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_hg += wp.tile_broadcast(wp.tile_load(bias_hg, shape=shape_b, offset=offset_b), shape=shape_T)
            gate_ho += wp.tile_broadcast(wp.tile_load(bias_ho, shape=shape_b, offset=offset_b), shape=shape_T)
        c = wp.tile_map(compute_cell_state, t_cell, gate_ii, gate_hi, gate_if, gate_hf, gate_ig, gate_hg)
        h = wp.tile_map(compute_hidden_state, c, gate_io, gate_ho)
        wp.tile_store(output_hidden, wp.tile_transpose(h), offset=offset)
        wp.tile_store(output_cell, wp.tile_transpose(c), offset=offset)

    return kernel


class LSTMCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a Long Short-Term Memory (LSTM) cell.

        .. math::

            \text{LSTMCell}(x, (h, c)) = (h', c')

        where

        .. math::

            \begin{array}{ll}
                i = \sigma(W_{ii} \, x + b_{ii} + W_{hi} \, h + b_{hi}) \\
                f = \sigma(W_{if} \, x + b_{if} + W_{hf} \, h + b_{hf}) \\
                g = \tanh(W_{ig} \, x + b_{ig} + W_{hg} \, h + b_{hg}) \\
                o = \sigma(W_{io} \, x + b_{io} + W_{ho} \, h + b_{ho}) \\
                c' = f \odot c + i \odot g \\
                h' = o \odot \tanh(c') \\
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
            * - :math:`W_{ii}, W_{if}, W_{ig}, W_{io}`
              - ``weight_ih``
              - ``(4 * hidden_size, input_size)``
              - Input-to-hidden weights
            * - :math:`W_{hi}, W_{hf}, W_{hg}, W_{ho}`
              - ``weight_hh``
              - ``(4 * hidden_size, hidden_size)``
              - Hidden-to-hidden weights
            * - :math:`b_{ii}, b_{if}, b_{ig}, b_{io}`
              - ``bias_ih``
              - ``(4 * hidden_size, 1)``
              - Input-to-hidden bias. Only if ``bias`` is true
            * - :math:`b_{hi}, b_{hf}, b_{hg}, b_{ho}`
              - ``bias_hh``
              - ``(4 * hidden_size, 1)``
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
            wp.empty(shape=(4 * self.hidden_size, self.input_size), dtype=wp.float32, device=self.device),
            requires_grad=self.requires_grad,
        )
        self.weight_hh = Parameter(
            wp.empty(shape=(4 * self.hidden_size, self.hidden_size), dtype=wp.float32, device=self.device),
            requires_grad=self.requires_grad,
        )
        self.register_parameter(name="weight_ih", parameter=self.weight_ih)
        self.register_parameter(name="weight_hh", parameter=self.weight_hh)
        # - biases
        if bias:
            self.bias_ih = Parameter(
                wp.empty(shape=(4 * self.hidden_size, 1), dtype=wp.float32, device=self.device),
                requires_grad=self.requires_grad,
            )
            self.bias_hh = Parameter(
                wp.empty(shape=(4 * self.hidden_size, 1), dtype=wp.float32, device=self.device),
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
            slice(3 * self.hidden_size, 4 * self.hidden_size),
        )

    def _initialize_parameters(self):
        kaiming_uniform(self.weight_ih.data, mode="scale", scale=1.0 / self.hidden_size)
        kaiming_uniform(self.weight_hh.data, mode="scale", scale=1.0 / self.hidden_size)
        if self.bias_ih:
            kaiming_uniform(self.bias_ih.data, mode="scale", scale=1.0 / self.hidden_size)
            kaiming_uniform(self.bias_hh.data, mode="scale", scale=1.0 / self.hidden_size)

    def __call__(self, input: wp.array, hidden: tuple[wp.array, wp.array]) -> tuple[wp.array, wp.array]:
        """Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, input_size)``.
        :param hidden: A tuple of the initial hidden state and cell state arrays, both with shapes ``(batch_size, hidden_size)``.

        :return: A tuple of the next hidden state and cell state arrays, both with shapes ``(batch_size, hidden_size)``.
        """
        dtype = input.dtype
        shape = (input.shape[0], self.hidden_size)
        key = (shape, dtype)
        # cache output
        if key not in self._cache:
            self._cache[key] = (
                wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad),
                wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad),
            )
        output = self._cache[key]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[
                input,
                hidden[0],
                hidden[1],
                self.weight_ih.data[self._slices[0]],
                self.weight_ih.data[self._slices[1]],
                self.weight_ih.data[self._slices[2]],
                self.weight_ih.data[self._slices[3]],
                self.weight_hh.data[self._slices[0]],
                self.weight_hh.data[self._slices[1]],
                self.weight_hh.data[self._slices[2]],
                self.weight_hh.data[self._slices[3]],
                self.bias_ih.data[self._slices[0]] if self.bias_ih else None,
                self.bias_ih.data[self._slices[1]] if self.bias_ih else None,
                self.bias_ih.data[self._slices[2]] if self.bias_ih else None,
                self.bias_ih.data[self._slices[3]] if self.bias_ih else None,
                self.bias_hh.data[self._slices[0]] if self.bias_hh else None,
                self.bias_hh.data[self._slices[1]] if self.bias_hh else None,
                self.bias_hh.data[self._slices[2]] if self.bias_hh else None,
                self.bias_hh.data[self._slices[3]] if self.bias_hh else None,
            ],
            outputs=[output[0], output[1]],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output
