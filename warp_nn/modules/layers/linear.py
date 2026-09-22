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

    @wp.kernel
    def kernel(
        input: wp.array2d[float],  # (batch_size, in_features)
        weight: wp.array2d[float],  # (out_features, in_features)
        bias: wp.array2d[float],  # (out_features, 1)
        output: wp.array2d[float],  # (batch_size, out_features)
    ):
        i, j = wp.tid()
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        out = wp.static(tile_transposed_gemm_2d(config.tile_2d))(weight, input, index=(i, j))
        if wp.static(include_bias):
            shape_T = (wp.static(config.tile_2d[1]), wp.static(config.tile_2d[0]))
            shape_b = (wp.static(config.tile_2d[1]), 1)
            offset_b = (j * wp.static(config.tile_2d[1]), 0)
            out += wp.tile_broadcast(wp.tile_load(bias, shape=shape_b, offset=offset_b), shape=shape_T)
        wp.tile_store(output, wp.tile_transpose(out), offset=offset)

    return kernel


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a linear transformation over the final dimension of the input.

        .. math::

            \text{Linear}(x) = W \, x + b

        |hr|

        Learnable parameters:

        .. list-table::
            :header-rows: 1

            * -
              - Name
              - Shape
              - Description
            * - :math:`W`
              - ``weight``
              - ``(out_features, in_features)``
              - Weights
            * - :math:`b`
              - ``bias``
              - ``(out_features, 1)``
              - Bias. Only if ``bias`` is true

        The parameters are initialized from the uniform distribution :math:`u(-k, k)`
        where :math:`k = \frac{1}{\sqrt{\text{in\_features}}}`.

        |hr|

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param bias: Whether to include a bias term.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=requires_grad)
        self.in_features = in_features
        self.out_features = out_features
        # create/register parameters
        # - weight
        shape = (self.out_features, self.in_features)
        self.weight = self.register_parameter(
            "weight",
            Parameter(wp.empty(shape=shape, dtype=wp.float32, device=self.device), requires_grad=self.requires_grad),
        )
        # - bias
        if bias:
            shape = (self.out_features, 1)
            self.bias = self.register_parameter(
                "bias",
                Parameter(
                    wp.empty(shape=shape, dtype=wp.float32, device=self.device), requires_grad=self.requires_grad
                ),
            )
        else:
            self.bias = None
        # set default/initial values
        if initialize_parameters:
            self._initialize_parameters()
        # runtime variables
        self._cache = {}
        self._config = get_kernel_config()
        self._kernel = _create_kernels(self._config, include_bias=self.bias is not None)

    def _initialize_parameters(self):
        kaiming_uniform(self.weight.data, mode="scale", scale=1.0 / self.in_features)
        if self.bias:
            kaiming_uniform(self.bias.data, mode="scale", scale=1.0 / self.in_features)

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(batch_size, in_features)``.

        :return: The output array, with shape ``(batch_size, out_features)``.
        """
        dtype = input.dtype
        shape = (input.shape[0], self.out_features)
        key = (shape, dtype)
        # cache output
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
        output = self._cache[key]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=resolve_dim(config=self._config, shape=shape, tiled=True),
            inputs=[input, self.weight.data, self.bias.data if self.bias else None],
            outputs=[output],
            device=self.device,
            block_dim=self._config.block_dim,
        )
        return output


class LazyLinear(Linear):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        *,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        """Apply a linear transformation whose ``in_features`` is inferred from the first input.

        See :py:class:`~warp_nn.modules.layers.linear.Linear` for details.

        :param out_features: The number of output features.
        :param bias: Whether to include a bias term.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        """
        # initialize the base class only, since the parameters are created on the first forward pass
        Module.__init__(self, requires_grad=requires_grad)
        self._out_features = out_features
        self._bias = bias
        self._initialize_parameters_on_build = initialize_parameters
        self._initialized = False

    def __call__(self, input: wp.array) -> wp.array:
        if not self._initialized:
            super().__init__(
                in_features=input.shape[1],
                out_features=self._out_features,
                bias=self._bias,
                initialize_parameters=self._initialize_parameters_on_build,
                requires_grad=self.requires_grad,
            )
            self._initialized = True
        return super().__call__(input)
