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
from warp_nn.utils import copy, get_kernel_config, resolve_dim

from .gru_cell import _create_kernels


class GRU(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        num_layers: int = 1,
        bidirectional: bool = False,
        bias: bool = True,
        initialize_parameters: bool = True,
        requires_grad: bool = True,
    ):
        r"""Apply a multi-layer Gated Recurrent Unit (GRU) to an input sequence.

        For each element in the input sequence, each layer computes

        .. math::

            \begin{array}{ll}
                r_t = \sigma(W_{ir} \, x_t + b_{ir} + W_{hr} \, h_{t-1} + b_{hr}) \\
                z_t = \sigma(W_{iz} \, x_t + b_{iz} + W_{hz} \, h_{t-1} + b_{hz}) \\
                n_t = \tanh(W_{in} \, x_t + b_{in} + r_t \odot (W_{hn} \, h_{t-1} + b_{hn})) \\
                h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
            \end{array}

        where :math:`h_t` is the hidden state at time :math:`t`, :math:`x_t` is the input at time :math:`t`
        (the output of the previous layer at time :math:`t` for the second and subsequent layers),
        and :math:`h_{t-1}` is the hidden state of the layer at time :math:`t-1` (or the initial hidden state
        at time :math:`0`), :math:`r_t`, :math:`z_t` and :math:`n_t` are the reset, update and new gates,
        :math:`\sigma` is the sigmoid function and :math:`\odot` is the element-wise product.

        If ``bidirectional`` is true, each layer also processes the sequence in reverse order (from the last element
        to the first one) with its own parameters, and the hidden states of both directions are concatenated
        (forward direction first) to form the output of the layer.

        |hr|

        Learnable parameters (where ``k`` is the layer index, and :math:`D = 2` if ``bidirectional`` is true,
        otherwise :math:`D = 1`):

        .. list-table::
            :header-rows: 1

            * -
              - Name
              - Shape
              - Description
            * - :math:`W_{ir}, W_{iz}, W_{in}`
              - ``weight_ih_l{k}``
              - ``(3 * hidden_size, input_size)`` for ``k = 0``, otherwise ``(3 * hidden_size, D * hidden_size)``
              - Input-to-hidden weights of the ``k``-th layer
            * - :math:`W_{hr}, W_{hz}, W_{hn}`
              - ``weight_hh_l{k}``
              - ``(3 * hidden_size, hidden_size)``
              - Hidden-to-hidden weights of the ``k``-th layer
            * - :math:`b_{ir}, b_{iz}, b_{in}`
              - ``bias_ih_l{k}``
              - ``(3 * hidden_size, 1)``
              - Input-to-hidden bias of the ``k``-th layer. Only if ``bias`` is true
            * - :math:`b_{hr}, b_{hz}, b_{hn}`
              - ``bias_hh_l{k}``
              - ``(3 * hidden_size, 1)``
              - Hidden-to-hidden bias of the ``k``-th layer. Only if ``bias`` is true

        The parameters of the reverse direction (only if ``bidirectional`` is true) are named the same,
        with the ``_reverse`` suffix (e.g. ``weight_ih_l{k}_reverse``).

        The parameters are initialized from the uniform distribution :math:`u(-k, k)`
        where :math:`k = \frac{1}{\sqrt{\text{hidden\_size}}}`.

        |hr|

        :param input_size: The number of input features.
        :param hidden_size: The number of hidden features.
        :param num_layers: The number of stacked recurrent layers.
        :param bidirectional: Whether each layer processes the sequence in both directions.
        :param bias: Whether to include a bias term.
        :param initialize_parameters: Whether to initialize the parameters with their default/initial values.
            If false, the parameters are left as uninitialized memory, and reading them (e.g. during a forward pass)
            before loading their values (e.g. from a state dictionary) is undefined behavior.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.

        :raises ValueError: If the number of layers is less than 1.
        """
        super().__init__(requires_grad=requires_grad)
        if num_layers < 1:
            raise ValueError(f"The number of layers must be greater than or equal to 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self._num_directions = 2 if bidirectional else 1
        # create/register parameters, per layer and direction
        # - list of {"weight_ih": ..., "weight_hh": ..., "bias_ih": ..., "bias_hh": ...},
        #   indexed by D * layer + direction
        self._layer_parameters = []
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self._num_directions * self.hidden_size
            for direction in range(self._num_directions):
                shapes = {
                    "weight_ih": (3 * self.hidden_size, layer_input_size),
                    "weight_hh": (3 * self.hidden_size, self.hidden_size),
                }
                if bias:
                    shapes |= {"bias_ih": (3 * self.hidden_size, 1), "bias_hh": (3 * self.hidden_size, 1)}
                parameters = {}
                for name, shape in shapes.items():
                    parameter = Parameter(
                        wp.empty(shape=shape, dtype=wp.float32, device=self.device),
                        requires_grad=self.requires_grad,
                    )
                    full_name = f"{name}_l{layer}{'_reverse' if direction else ''}"
                    parameters[name] = self.register_parameter(name=full_name, parameter=parameter)
                    setattr(self, full_name, parameter)
                self._layer_parameters.append(parameters)
        # set default/initial values
        if initialize_parameters:
            self._initialize_parameters()
        # runtime variables
        self._config = get_kernel_config()
        self._kernel = _create_kernels(self._config, include_bias=bias)
        self._slices = (
            slice(0 * self.hidden_size, 1 * self.hidden_size),
            slice(1 * self.hidden_size, 2 * self.hidden_size),
            slice(2 * self.hidden_size, 3 * self.hidden_size),
        )

    def _initialize_parameters(self):
        for parameter in self.parameters():
            kaiming_uniform(parameter, mode="scale", scale=1.0 / self.hidden_size)

    def __call__(self, input: wp.array, hidden: wp.array | None = None) -> tuple[wp.array, wp.array]:
        """Forward pass of the module.

        :param input: The input sequence array, with shape ``(batch_size, sequence_length, input_size)``.
        :param hidden: The initial hidden state array, with shape ``(D * num_layers, batch_size, hidden_size)``.
            If not given, it defaults to zeros.

        :return: A tuple of the output sequence array (the hidden states of the last layer for each element in the
            sequence), with shape ``(batch_size, sequence_length, D * hidden_size)``, and the final hidden state
            array (of each layer and direction), with shape ``(D * num_layers, batch_size, hidden_size)``.

        :raises ValueError: If the shape of the input or hidden state array is not valid.
        """
        if input.ndim != 3 or input.shape[2] != self.input_size:
            raise ValueError(
                f"Expected an input array with shape (batch_size, sequence_length, {self.input_size}), "
                f"got {tuple(input.shape)}"
            )
        batch_size, sequence_length, _ = input.shape
        num_directions, hidden_size = self._num_directions, self.hidden_size
        hidden_shape = (num_directions * self.num_layers, batch_size, hidden_size)
        if hidden is not None and tuple(hidden.shape) != hidden_shape:
            raise ValueError(f"Expected a hidden state array with shape {hidden_shape}, got {tuple(hidden.shape)}")
        dtype = input.dtype
        key = (tuple(input.shape), dtype)
        # cache outputs (of each layer), final hidden state and (default) initial hidden state
        if key not in self._cache:
            output_shape = (batch_size, sequence_length, num_directions * hidden_size)
            self._cache[key] = (
                [
                    wp.empty(output_shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
                    for _ in range(self.num_layers)
                ],
                wp.empty(hidden_shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad),
                wp.zeros(hidden_shape, dtype=dtype, device=self.device),
            )
        outputs, output_hidden, initial_hidden = self._cache[key]
        if hidden is None:
            hidden = initial_hidden
        # launch kernels, one per layer, direction and time step
        # - each time step writes its hidden state to its own slice of the layer output array (rather than
        #   overwriting the same array), so that the tape can propagate the gradients through the sequence
        dim = resolve_dim(config=self._config, shape=(batch_size, hidden_size), tiled=True)
        layer_input = input
        for layer, layer_output in enumerate(outputs):
            for direction in range(num_directions):
                index = layer * num_directions + direction
                parameters = self._layer_parameters[index]
                # split the parameters into the reset, update and new gates' parameters
                weights = [parameters[name].data[s] for name in ("weight_ih", "weight_hh") for s in self._slices]
                if "bias_ih" in parameters:
                    biases = [parameters[name].data[s] for name in ("bias_ih", "bias_hh") for s in self._slices]
                else:
                    biases = [None] * 6
                features = slice(direction * hidden_size, (direction + 1) * hidden_size)
                state = hidden[index]
                for t in reversed(range(sequence_length)) if direction else range(sequence_length):
                    output = layer_output[:, t, features]
                    wp.launch_tiled(
                        self._kernel,
                        dim=dim,
                        inputs=[layer_input[:, t], state, *weights, *biases],
                        outputs=[output],
                        device=self.device,
                        block_dim=self._config.block_dim,
                    )
                    state = output
                copy(state, output_hidden[index])
            layer_input = layer_output
        return outputs[-1], output_hidden
