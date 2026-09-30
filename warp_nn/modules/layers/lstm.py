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

from .lstm_cell import _create_kernels


class LSTM(Module):
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
        r"""Apply a multi-layer Long Short-Term Memory (LSTM) to an input sequence.

        For each element in the input sequence, each layer computes

        .. math::

            \begin{array}{ll}
                i_t = \sigma(W_{ii} \, x_t + b_{ii} + W_{hi} \, h_{t-1} + b_{hi}) \\
                f_t = \sigma(W_{if} \, x_t + b_{if} + W_{hf} \, h_{t-1} + b_{hf}) \\
                g_t = \tanh(W_{ig} \, x_t + b_{ig} + W_{hg} \, h_{t-1} + b_{hg}) \\
                o_t = \sigma(W_{io} \, x_t + b_{io} + W_{ho} \, h_{t-1} + b_{ho}) \\
                c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
                h_t = o_t \odot \tanh(c_t)
            \end{array}

        where :math:`h_t` and :math:`c_t` are the hidden and cell states at time :math:`t`, :math:`x_t` is the input
        at time :math:`t`
        (the output of the previous layer at time :math:`t` for the second and subsequent layers),
        :math:`h_{t-1}` and :math:`c_{t-1}` are the hidden and cell states of the layer at time :math:`t-1`
        (or the initial hidden and cell states at time :math:`0`), :math:`i_t`, :math:`f_t`, :math:`g_t` and
        :math:`o_t` are the input, forget, cell and output gates,
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
            * - :math:`W_{ii}, W_{if}, W_{ig}, W_{io}`
              - ``weight_ih_l{k}``
              - ``(4 * hidden_size, input_size)`` for ``k = 0``, otherwise ``(4 * hidden_size, D * hidden_size)``
              - Input-to-hidden weights of the ``k``-th layer
            * - :math:`W_{hi}, W_{hf}, W_{hg}, W_{ho}`
              - ``weight_hh_l{k}``
              - ``(4 * hidden_size, hidden_size)``
              - Hidden-to-hidden weights of the ``k``-th layer
            * - :math:`b_{ii}, b_{if}, b_{ig}, b_{io}`
              - ``bias_ih_l{k}``
              - ``(4 * hidden_size, 1)``
              - Input-to-hidden bias of the ``k``-th layer. Only if ``bias`` is true
            * - :math:`b_{hi}, b_{hf}, b_{hg}, b_{ho}`
              - ``bias_hh_l{k}``
              - ``(4 * hidden_size, 1)``
              - Hidden-to-hidden bias of the ``k``-th layer. Only if ``bias`` is true

        The parameters of the reverse direction (only if ``bidirectional`` is true) are named the same,
        with the ``_reverse`` suffix (e.g. ``weight_ih_l{k}_reverse``).

        The parameters are initialized from the uniform distribution :math:`u(-k, k)`
        where :math:`k = \frac{1}{\sqrt{\text{hidden\_size}}}`.

        .. note::

            On CUDA, the backward pass (i.e. the gradient computation) of the default (32, 32) tile shape requires
            more shared memory than available on some devices. In that case, create the module with a smaller tile
            shape, e.g. within a ``warp_nn.utils.kernel_config(tile_2d=(16, 16))`` context.

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
                    "weight_ih": (4 * self.hidden_size, layer_input_size),
                    "weight_hh": (4 * self.hidden_size, self.hidden_size),
                }
                if bias:
                    shapes |= {"bias_ih": (4 * self.hidden_size, 1), "bias_hh": (4 * self.hidden_size, 1)}
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
            slice(3 * self.hidden_size, 4 * self.hidden_size),
        )

    def _initialize_parameters(self):
        for parameter in self.parameters():
            kaiming_uniform(parameter, mode="scale", scale=1.0 / self.hidden_size)

    def __call__(
        self, input: wp.array, hidden: tuple[wp.array, wp.array] | None = None
    ) -> tuple[wp.array, tuple[wp.array, wp.array]]:
        """Forward pass of the module.

        :param input: The input sequence array, with shape ``(batch_size, sequence_length, input_size)``.
        :param hidden: A tuple of the initial hidden state and cell state arrays, both with shapes
            ``(D * num_layers, batch_size, hidden_size)``. If not given, they default to zeros.

        :return: A tuple of the output sequence array (the hidden states of the last layer for each element in the
            sequence), with shape ``(batch_size, sequence_length, D * hidden_size)``, and a tuple of the final
            hidden state and cell state arrays (of each layer and direction), both with shapes
            ``(D * num_layers, batch_size, hidden_size)``.

        :raises ValueError: If the shape of the input, hidden state or cell state array is not valid,
            or if the hidden and cell states are not given as a tuple.
        """
        if input.ndim != 3 or input.shape[2] != self.input_size:
            raise ValueError(
                f"Expected an input array with shape (batch_size, sequence_length, {self.input_size}), "
                f"got {tuple(input.shape)}"
            )
        batch_size, sequence_length, _ = input.shape
        num_directions, hidden_size = self._num_directions, self.hidden_size
        hidden_shape = (num_directions * self.num_layers, batch_size, hidden_size)
        if hidden is not None:
            if not isinstance(hidden, tuple) or len(hidden) != 2:
                raise ValueError("Expected a tuple of the initial hidden state and cell state arrays")
            for name, array in zip(("hidden state", "cell state"), hidden):
                if tuple(array.shape) != hidden_shape:
                    raise ValueError(f"Expected a {name} array with shape {hidden_shape}, got {tuple(array.shape)}")
        dtype = input.dtype
        key = (tuple(input.shape), dtype)
        # cache outputs (of each layer), cell states (of each layer, direction and time step),
        # final hidden and cell states and (default) initial hidden and cell states
        if key not in self._cache:
            output_shape = (batch_size, sequence_length, num_directions * hidden_size)
            cell_shape = (num_directions * self.num_layers, sequence_length, batch_size, hidden_size)
            self._cache[key] = (
                [
                    wp.empty(output_shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad)
                    for _ in range(self.num_layers)
                ],
                wp.empty(cell_shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad),
                (
                    wp.empty(hidden_shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad),
                    wp.empty(hidden_shape, dtype=dtype, device=self.device, requires_grad=self.requires_grad),
                ),
                (
                    wp.zeros(hidden_shape, dtype=dtype, device=self.device),
                    wp.zeros(hidden_shape, dtype=dtype, device=self.device),
                ),
            )
        outputs, cells, output_hidden, initial_hidden = self._cache[key]
        if hidden is None:
            hidden = initial_hidden
        # launch kernels, one per layer, direction and time step
        # - each time step writes its hidden and cell states to their own slices of the layer output and cell state
        #   arrays (rather than overwriting the same arrays), so that the tape can propagate the gradients through
        #   the sequence
        dim = resolve_dim(config=self._config, shape=(batch_size, hidden_size), tiled=True)
        layer_input = input
        for layer, layer_output in enumerate(outputs):
            for direction in range(num_directions):
                index = layer * num_directions + direction
                parameters = self._layer_parameters[index]
                # split the parameters into the input, forget, cell and output gates' parameters
                weights = [parameters[name].data[s] for name in ("weight_ih", "weight_hh") for s in self._slices]
                if "bias_ih" in parameters:
                    biases = [parameters[name].data[s] for name in ("bias_ih", "bias_hh") for s in self._slices]
                else:
                    biases = [None] * 8
                features = slice(direction * hidden_size, (direction + 1) * hidden_size)
                state, cell = hidden[0][index], hidden[1][index]
                for t in reversed(range(sequence_length)) if direction else range(sequence_length):
                    output, output_cell = layer_output[:, t, features], cells[index, t]
                    wp.launch_tiled(
                        self._kernel,
                        dim=dim,
                        inputs=[layer_input[:, t], state, cell, *weights, *biases],
                        outputs=[output, output_cell],
                        device=self.device,
                        block_dim=self._config.block_dim,
                    )
                    state, cell = output, output_cell
                copy(state, output_hidden[0][index])
                copy(cell, output_hidden[1][index])
            layer_input = layer_output
        return outputs[-1], output_hidden
