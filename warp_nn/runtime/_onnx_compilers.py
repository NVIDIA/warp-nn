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

"""Compilation of ONNX graph nodes into operations executed by Warp-NN modules (used by ``OnnxRuntime``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warp_nn import nn
from warp_nn.runtime._onnx_kernels import batch_first_to_onnx_sequence, sequence_first_to_batch_first


if TYPE_CHECKING:
    from warp_nn.runtime.onnx_runtime import _Node


_Operand = str | np.ndarray  # name of a runtime array in the tensor table, or constant value
_Compiler = Callable[["_Node", "CompilationContext"], "_Operation"]


# Compilation helpers


@dataclass(kw_only=True)
class CompilationContext:
    """Model data shared by node compilers."""

    device: wp.Device
    requires_grad: bool
    opset: int
    constants: dict[str, np.ndarray]
    # constants read as runtime arrays (rather than being loaded into modules), which must be uploaded to the device
    runtime_constants: set[str] = field(default_factory=set)

    @staticmethod
    def input_name(node: _Node, index: int) -> str:
        """Get the name of a node input, or an empty name if it is absent."""
        return node.inputs[index] if index < len(node.inputs) else ""

    def constant(self, node: _Node, index: int, *, required: bool = True) -> np.ndarray | None:
        """Get the value of a node input that must be constant, or None if it is absent and optional."""
        name = self.input_name(node, index)
        if not name:
            if required:
                raise ValueError(f"OnnxRuntime {node.op_type}: input {index} is required")
            return None
        if name not in self.constants:
            raise NotImplementedError(
                f"OnnxRuntime {node.op_type}: input '{name}' must be a constant (initializer or Constant node)"
            )
        return self.constants[name]

    def tensor(self, node: _Node, index: int, *, required: bool = True) -> str | None:
        """Get the name of a node input read at runtime, or None if it is absent and optional."""
        name = self.input_name(node, index)
        if not name:
            if required:
                raise ValueError(f"OnnxRuntime {node.op_type}: input {index} is required")
            return None
        if name in self.constants:
            self.runtime_constants.add(name)
        return name

    def operand(self, node: _Node, index: int) -> _Operand:
        """Get a required element-wise operand: its constant value if known, or its name otherwise."""
        name = self.input_name(node, index)
        if not name:
            raise ValueError(f"OnnxRuntime {node.op_type}: input {index} is required")
        return self.constants.get(name, name)


def _attributes(node: _Node, **defaults: Any) -> dict[str, Any]:
    """Get the node attributes (or their defaults), rejecting any attribute that is not listed."""
    if unsupported := set(node.attributes) - set(defaults):
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: unsupported attributes {sorted(unsupported)}")
    return {name: node.attributes.get(name, default) for name, default in defaults.items()}


def _check_outputs(node: _Node, count: int) -> None:
    """Reject the node if any optional output beyond the first ``count`` ones is used."""
    if any(node.outputs[count:]):
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: only the first {count} output(s) are supported")


def _load_state(module: nn.Module, **values: np.ndarray) -> None:
    """Load constant values into a module's parameters/buffers, checking their shapes."""
    state = module.state_dict()
    for name, value in values.items():
        expected = tuple(state[name].shape)
        if np.shape(value) != expected:
            raise ValueError(
                f"OnnxRuntime: '{name}' of module '{type(module).__name__}' requires shape {expected}, "
                f"got {np.shape(value)}"
            )
    module.load_state_dict({name: np.ascontiguousarray(value, dtype=np.float32) for name, value in values.items()})


def _padding(node: _Node, auto_pad: str, pads: tuple[int, ...] | None, spatial_dims: int) -> tuple[int, ...]:
    """Convert ONNX (begin..., end...) pads to the symmetric padding supported by the modules."""
    if auto_pad not in ("NOTSET", "VALID"):
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: auto_pad '{auto_pad}' is not supported")
    if auto_pad == "VALID" or pads is None:
        return (0,) * spatial_dims
    begin, end = tuple(pads[:spatial_dims]), tuple(pads[spatial_dims:])
    if begin != end:
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: only symmetric pads are supported, got {pads}")
    return begin


def _spatial_dims(node: _Node, kernel_shape: tuple[int, ...]) -> int:
    """Get the number of spatial dimensions of a convolution/pooling node, which must be 1 or 2."""
    if len(kernel_shape) not in (1, 2):
        raise NotImplementedError(
            f"OnnxRuntime {node.op_type}: only 1D and 2D kernels are supported, got kernel shape {kernel_shape}"
        )
    return len(kernel_shape)


def _broadcast_view(array: wp.array, shape: tuple[int, ...]) -> wp.array:
    """Create a (non-contiguous) view of an array broadcast to a shape, with zero strides in broadcast dimensions."""
    offset = len(shape) - array.ndim
    strides = tuple(
        0 if i < offset or array.shape[i - offset] != size else array.strides[i - offset]
        for i, size in enumerate(shape)
    )
    return wp.array(ptr=array.ptr, shape=shape, strides=strides, dtype=array.dtype, device=array.device)


# Operations


class _Operation(ABC):
    """Graph node executed by Warp-NN modules."""

    node: _Node

    @property
    @abstractmethod
    def modules(self) -> tuple[nn.Module, ...]:
        """Modules executing the operation."""

    @abstractmethod
    def __call__(self, tensors: dict[str, wp.array]) -> None:
        """Read the inputs from, and write the outputs to, the tensor table."""


@dataclass(eq=False, kw_only=True)
class _ModuleOperation(_Operation):
    """Call a module (or a function wrapping it) with named runtime inputs and register its outputs."""

    node: _Node
    module: nn.Module
    inputs: tuple[str, ...]
    forward: Callable[..., wp.array | tuple[wp.array, ...]] | None = None

    @property
    def modules(self) -> tuple[nn.Module, ...]:
        return (self.module,)

    def __call__(self, tensors: dict[str, wp.array]) -> None:
        forward = self.module if self.forward is None else self.forward
        outputs = forward(*(tensors[name] for name in self.inputs))
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for name, output in zip(self.node.outputs, outputs):
            tensors[name] = output


@dataclass(eq=False, kw_only=True)
class _ReshapeOperation(_Operation):
    """Reshape a runtime array into a view (no data is copied, and no kernel is launched)."""

    node: _Node
    input: str
    output_shape: Callable[[tuple[int, ...]], tuple[int, ...]]  # maps the input shape to the output shape

    @property
    def modules(self) -> tuple[nn.Module, ...]:
        return ()

    def __call__(self, tensors: dict[str, wp.array]) -> None:
        input = tensors[self.input]
        tensors[self.node.outputs[0]] = input.reshape(self.output_shape(tuple(input.shape)))


@dataclass(eq=False, kw_only=True)
class _ElementwiseOperation(_Operation):
    """Call element-wise modules with multidirectional (NumPy-style) broadcasting of their operands.

    Element-wise modules require 1D to 3D operands with the same shape and data type. Therefore, operands are
    broadcast into cached arrays (once per shape for constants, which are also converted to the data type of the
    runtime operands, and at every call for runtime arrays), and arrays with more than 3 dimensions are processed
    as flat views.

    Unary and binary operators are executed by a single module. Variadic operators (e.g. ``Max``) with N operands
    are executed by a chain of N - 1 binary modules, which fold the operands from left to right. Each module owns
    its (cached) output array, so no module reads the array it writes.
    """

    node: _Node
    chain: tuple[nn.Module, ...]
    operands: tuple[_Operand, ...]
    _cache: dict[tuple[int, tuple[int, ...], type], wp.array] = field(default_factory=dict, init=False, repr=False)

    @property
    def modules(self) -> tuple[nn.Module, ...]:
        return self.chain

    def __call__(self, tensors: dict[str, wp.array]) -> None:
        values = tuple(tensors[operand] if isinstance(operand, str) else operand for operand in self.operands)
        dtypes = {value.dtype for value in values if isinstance(value, wp.array)}
        if len(dtypes) > 1:
            names = ", ".join(sorted(dtype.__name__ for dtype in dtypes))
            raise NotImplementedError(
                f"OnnxRuntime {self.node.op_type}: runtime operands with different dtypes ({names}) are not supported"
            )
        (dtype,) = dtypes
        shapes = tuple(tuple(value.shape) for value in values)
        try:
            shape = tuple(np.broadcast_shapes(*shapes))
        except ValueError as exc:
            raise ValueError(f"OnnxRuntime {self.node.op_type}: operand shapes {shapes} are not broadcastable") from exc
        arrays = tuple(self._broadcast(index, value, shape, dtype) for index, value in enumerate(values))
        if len(shape) > 3:
            arrays = tuple(array.reshape((math.prod(shape),)) for array in arrays)
        if len(arrays) == 1:
            output = self.chain[0](arrays[0])
        else:
            output = arrays[0]
            for module, array in zip(self.chain, arrays[1:]):
                output = module(output, array)
        tensors[self.node.outputs[0]] = output.reshape(shape) if len(shape) > 3 else output

    def _broadcast(self, index: int, value: wp.array | np.ndarray, shape: tuple[int, ...], dtype: type) -> wp.array:
        if isinstance(value, wp.array) and tuple(value.shape) == shape:
            return value
        key = (index, shape, dtype)
        if isinstance(value, np.ndarray):
            if key not in self._cache:
                self._cache[key] = wp.array(
                    np.broadcast_to(value, shape).astype(wp.dtype_to_numpy(dtype)),
                    dtype=dtype,
                    device=self.chain[0].device,
                )
            return self._cache[key]
        # the copy from the broadcast view would not accumulate the gradients over the broadcast dimensions
        if self.chain[0].requires_grad:
            raise NotImplementedError(
                f"OnnxRuntime {self.node.op_type}: broadcasting runtime operands is not supported with gradients"
            )
        if key not in self._cache:
            self._cache[key] = wp.empty(shape, dtype=dtype, device=self.chain[0].device)
        output = self._cache[key]
        wp.copy(output, _broadcast_view(value, shape))
        return output


@dataclass(eq=False, kw_only=True)
class _RecurrentOperation(_Operation):
    """Execute an ONNX ``RNN``, ``GRU`` or ``LSTM`` node with a single-layer recurrent module.

    The module processes batch-first sequences, so the input is converted from the ONNX layout
    ``(seq_length, batch_size, input_size)`` and the output sequence ``Y`` (only if it is used) is converted to
    ``(seq_length, num_directions, batch_size, hidden_size)``, by differentiable kernels. The reverse direction
    (without the forward one) is executed by the forward module, reversing the time steps of the input and output.
    The final states (``Y_h`` and, for LSTM, ``Y_c``) are the ones returned by the module.
    """

    node: _Node
    module: nn.Module
    flip: bool  # whether the sequence is processed in reverse order by a unidirectional module
    input: str
    initial_states: tuple[str | None, ...]  # initial_h (and initial_c, for LSTM)
    _cache: dict[tuple[int, int], tuple[wp.array, wp.array | None, tuple[wp.array, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def modules(self) -> tuple[nn.Module, ...]:
        return (self.module,)

    def __call__(self, tensors: dict[str, wp.array]) -> None:
        module = self.module
        input = tensors[self.input]
        input_size, hidden_size = module.input_size, module.hidden_size
        if input.ndim != 3 or input.shape[2] != input_size:
            raise ValueError(
                f"OnnxRuntime {self.node.op_type}: expected an input with shape "
                f"(seq_length, batch_size, {input_size}), got {tuple(input.shape)}"
            )
        seq_length, batch_size = input.shape[:2]
        num_directions = 2 if module.bidirectional else 1
        state_shape = (num_directions, batch_size, hidden_size)
        initial_states = tuple(None if name is None else tensors[name] for name in self.initial_states)
        for name, initial_state in zip(self.initial_states, initial_states):
            if initial_state is not None and tuple(initial_state.shape) != state_shape:
                raise ValueError(
                    f"OnnxRuntime {self.node.op_type}: initial state '{name}' must have shape {state_shape}, "
                    f"got {tuple(initial_state.shape)}"
                )
        # cache the batch-first input, output sequence (only if used) and default (zero) initial states
        key = (seq_length, batch_size)
        if key not in self._cache:
            arguments = {"dtype": wp.float32, "device": module.device}
            grad = {"requires_grad": module.requires_grad}
            batch_first = wp.empty((batch_size, seq_length, input_size), **arguments, **grad)
            sequence = None
            if self.node.outputs[0]:
                sequence = wp.empty((seq_length, num_directions, batch_size, hidden_size), **arguments, **grad)
            zeros = tuple(wp.zeros(state_shape, **arguments) for _ in self.initial_states)
            self._cache[key] = (batch_first, sequence, zeros)
        batch_first, sequence, zeros = self._cache[key]
        reverse = int(self.flip)
        wp.launch(
            sequence_first_to_batch_first,
            dim=tuple(input.shape),
            inputs=[input, reverse],
            outputs=[batch_first],
            device=module.device,
        )
        hidden = None
        if any(state is not None for state in initial_states):
            hidden = tuple(zero if state is None else state for state, zero in zip(initial_states, zeros))
            hidden = hidden if len(hidden) > 1 else hidden[0]
        output, final_states = module(batch_first, hidden)
        if not isinstance(final_states, tuple):
            final_states = (final_states,)
        if sequence is not None:
            wp.launch(
                batch_first_to_onnx_sequence,
                dim=tuple(sequence.shape),
                inputs=[output, reverse],
                outputs=[sequence],
                device=module.device,
            )
        for name, array in zip(self.node.outputs, (sequence, *final_states)):
            if name:
                tensors[name] = array


# Compiler factories


def _elementwise_operation(
    node: _Node, context: CompilationContext, chain: tuple[nn.Module, ...], arity: int
) -> _ElementwiseOperation:
    """Create an element-wise operation executed by a chain of modules, whose operands are the first ``arity``
    node inputs."""
    _check_outputs(node, 1)
    operands = tuple(context.operand(node, index) for index in range(arity))
    if all(isinstance(operand, np.ndarray) for operand in operands):
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: operators with only constant inputs are not supported")
    return _ElementwiseOperation(node=node, chain=chain, operands=operands)


def _elementwise(
    module_type: type[nn.Module],
    *,
    arity: int = 1,
    differentiable: bool = True,
    attributes: dict[str, tuple[str, Any]] | None = None,
    **arguments: Any,
) -> _Compiler:
    """Create a compiler for an element-wise operator executed by a module of the given type.

    :param module_type: The module type.
    :param arity: The number of operands.
    :param differentiable: Whether the module type accepts the ``requires_grad`` constructor argument.
    :param attributes: Constructor arguments taken from the node attributes, mapped to the ONNX attribute names
        and default values: ``{argument: (attribute, default)}``.
    :param arguments: Fixed constructor arguments.
    """
    attributes = attributes or {}

    def compile(node: _Node, context: CompilationContext) -> _Operation:
        values = _attributes(node, **dict(attributes.values()))
        kwargs = {argument: values[name] for argument, (name, _) in attributes.items()} | arguments
        if differentiable:
            kwargs["requires_grad"] = context.requires_grad
        return _elementwise_operation(node, context, (module_type(**kwargs),), arity)

    return compile


def _variadic(module_type: type[nn.Module]) -> _Compiler:
    """Create a compiler for a variadic element-wise operator (e.g. ``Max``) executed by binary modules.

    :param module_type: The binary module type, which is applied to fold the operands from left to right.
    """

    def compile(node: _Node, context: CompilationContext) -> _Operation:
        _attributes(node)
        if len(node.inputs) == 1:
            return _compile_identity(node, context)
        chain = tuple(module_type(requires_grad=context.requires_grad) for _ in node.inputs[1:])
        return _elementwise_operation(node, context, chain, len(node.inputs))

    return compile


# Operator compilers (sorted by ONNX operator name)


def _compile_pool(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``AveragePool``/``MaxPool`` node with a 1D or 2D kernel into a pooling module."""
    common = {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "dilations": None,
        "kernel_shape": None,
        "pads": None,
        "strides": None,
    }
    if node.op_type == "MaxPool":
        _check_outputs(node, 1)
        attributes = _attributes(node, **common, storage_order=0)  # the storage order only affects the indices
    else:
        attributes = _attributes(node, **common, count_include_pad=0)
    spatial_dims = _spatial_dims(node, tuple(attributes["kernel_shape"]))
    arguments = {
        "kernel_size": tuple(attributes["kernel_shape"]),
        "stride": tuple(attributes["strides"] or (1,) * spatial_dims),
        "padding": _padding(node, attributes["auto_pad"], attributes["pads"], spatial_dims),
        "ceil_mode": bool(attributes["ceil_mode"]),
        "requires_grad": context.requires_grad,
    }
    dilation = tuple(attributes["dilations"] or (1,) * spatial_dims)
    if node.op_type == "MaxPool":
        module = (nn.MaxPool1D if spatial_dims == 1 else nn.MaxPool2D)(**arguments, dilation=dilation)
    else:
        if any(d != 1 for d in dilation):
            raise NotImplementedError(f"OnnxRuntime AveragePool: dilations are not supported, got {dilation}")
        count_include_pad = bool(attributes["count_include_pad"])
        module = (nn.AvgPool1D if spatial_dims == 1 else nn.AvgPool2D)(**arguments, count_include_pad=count_include_pad)
    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),))


def _compile_batch_norm(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``BatchNormalization`` node (inference mode) into a :class:`BatchNorm` module."""
    _check_outputs(node, 1)
    attributes = _attributes(node, epsilon=1e-5, momentum=0.9, spatial=1, training_mode=0)
    if attributes["training_mode"] or not attributes["spatial"]:
        raise NotImplementedError("OnnxRuntime BatchNormalization: only spatial inference mode is supported")
    scale, bias, mean, var = (context.constant(node, index) for index in range(1, 5))
    module = nn.BatchNorm(
        scale.shape[0], eps=attributes["epsilon"], initialize_parameters=False, requires_grad=context.requires_grad
    )
    _load_state(module, weight=scale, bias=bias, running_mean=mean, running_var=var)
    return _ModuleOperation(node=node, module=module.eval(), inputs=(context.tensor(node, 0),))


def _compile_clip(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Clip`` node, whose bounds are attributes (opset < 11) or constant inputs."""
    if context.opset < 11:
        bounds = tuple(_attributes(node, min=None, max=None).values())
    else:
        _attributes(node)
        bounds = tuple(context.constant(node, index, required=False) for index in (1, 2))
    min_val, max_val = (None if bound is None else float(np.asarray(bound).item()) for bound in bounds)
    return _elementwise_operation(node, context, (nn.Clip(min_val, max_val, requires_grad=context.requires_grad),), 1)


def _compile_conv(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Conv`` node with a 1D or 2D kernel into a :class:`Conv1D`/:class:`Conv2D` module."""
    attributes = _attributes(
        node, auto_pad="NOTSET", dilations=None, group=1, kernel_shape=None, pads=None, strides=None
    )
    weight = context.constant(node, 1)
    bias = context.constant(node, 2, required=False)
    spatial_dims = _spatial_dims(node, weight.shape[2:])
    if attributes["kernel_shape"] is not None and tuple(attributes["kernel_shape"]) != weight.shape[2:]:
        raise ValueError(
            f"OnnxRuntime Conv: kernel_shape {attributes['kernel_shape']} does not match weight shape {weight.shape}"
        )
    groups = attributes["group"]
    in_channels = weight.shape[1] * groups
    module_type = nn.Conv1D if spatial_dims == 1 else nn.Conv2D
    module = module_type(
        in_channels,
        weight.shape[0],
        tuple(weight.shape[2:]),
        stride=tuple(attributes["strides"] or (1,) * spatial_dims),
        padding=_padding(node, attributes["auto_pad"], attributes["pads"], spatial_dims),
        dilation=tuple(attributes["dilations"] or (1,) * spatial_dims),
        groups=groups,
        bias=bias is not None,
        initialize_parameters=False,
        requires_grad=context.requires_grad,
    )
    if bias is None:
        _load_state(module, weight=weight)
    else:
        _load_state(module, weight=weight, bias=np.reshape(bias, (-1, 1)))

    def forward(input: wp.array) -> wp.array:
        if input.ndim != spatial_dims + 2 or input.shape[1] != in_channels:
            raise ValueError(
                f"OnnxRuntime Conv: expected an input with shape (batch_size, {in_channels}, *spatial) "
                f"with {spatial_dims} spatial dimension(s), got {tuple(input.shape)}"
            )
        return module(input)

    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),), forward=forward)


def _compile_dropout(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Dropout`` node (inference mode) into a :class:`Dropout` module in evaluation mode."""
    _check_outputs(node, 1)
    if context.opset < 12:
        ratio = _attributes(node, ratio=0.5, is_test=1)["ratio"]
    else:
        _attributes(node, seed=0)
        ratio = context.constant(node, 1, required=False)
        training_mode = context.constant(node, 2, required=False)
        if training_mode is not None and training_mode.item():
            raise NotImplementedError("OnnxRuntime Dropout: training mode is not supported")
    p = 0.5 if ratio is None else float(np.asarray(ratio).item())
    return _ModuleOperation(
        node=node, module=nn.Dropout(p, requires_grad=context.requires_grad).eval(), inputs=(context.tensor(node, 0),)
    )


def _compile_flatten(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Flatten`` node, which flattens the input into 2D, into a :class:`Flatten` module."""
    axis = _attributes(node, axis=1)["axis"]
    module = nn.Flatten(start_dim=axis)

    def forward(input: wp.array) -> wp.array:
        shape = tuple(input.shape)
        start = axis + len(shape) if axis < 0 else axis
        output_shape = (math.prod(shape[:start]), math.prod(shape[start:]))
        # the module flattens the dimensions from axis onward, and the leading ones are flattened by the reshape
        output = input if start == len(shape) else module(input)
        return output.reshape(output_shape)

    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),), forward=forward)


def _compile_linear(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Gemm`` (Y = alpha * A @ B + beta * C) or ``MatMul`` node with a constant ``B``.

    ``MatMul`` inputs with more (or less) than 2 dimensions are flattened into a batch of rows.
    """
    if node.op_type == "Gemm":
        attributes = _attributes(node, alpha=1.0, beta=1.0, transA=0, transB=0)
        if attributes["transA"]:
            raise NotImplementedError("OnnxRuntime Gemm: transA is not supported")
        bias = context.constant(node, 2, required=False)
    else:
        attributes = {"alpha": 1.0, "beta": 1.0, "transB": 0}
        _attributes(node)
        bias = None
    weight = context.constant(node, 1)
    if weight.ndim != 2:
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: B must be 2D, got shape {weight.shape}")
    weight = attributes["alpha"] * (weight if attributes["transB"] else weight.T)
    out_features, in_features = weight.shape

    module = nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        initialize_parameters=False,
        requires_grad=context.requires_grad,
    )
    if bias is None:
        _load_state(module, weight=weight)
    else:
        try:
            bias = np.broadcast_to(bias, (1, out_features)).reshape(out_features, 1)
        except ValueError as exc:
            raise NotImplementedError(
                f"OnnxRuntime Gemm: C must be broadcastable to (1, {out_features}), got shape {np.shape(bias)}"
            ) from exc
        _load_state(module, weight=weight, bias=attributes["beta"] * bias)

    def forward(input: wp.array) -> wp.array:
        shape = tuple(input.shape)
        if shape[-1] != in_features:
            raise ValueError(f"OnnxRuntime {node.op_type}: expected {in_features} input features, got shape {shape}")
        if len(shape) == 2:
            return module(input)
        rows = math.prod(shape[:-1])
        return module(input.reshape((rows, in_features))).reshape(shape[:-1] + (out_features,))

    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),), forward=forward)


def _compile_global_pool(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``GlobalAveragePool``/``GlobalMaxPool`` node into a global pooling module."""
    _attributes(node)
    module_type = nn.GlobalAvgPool if node.op_type == "GlobalAveragePool" else nn.GlobalMaxPool
    return _ModuleOperation(
        node=node, module=module_type(requires_grad=context.requires_grad), inputs=(context.tensor(node, 0),)
    )


def _compile_group_norm(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``GroupNormalization`` node into a :class:`GroupNorm` module.

    The ONNX checker rejects the deprecated opset 18 version (with per-group scale and bias), so the scale and bias
    are per-channel.
    """
    attributes = _attributes(node, epsilon=1e-5, num_groups=None, stash_type=1)
    scale, bias = context.constant(node, 1), context.constant(node, 2)
    module = nn.GroupNorm(
        attributes["num_groups"],
        scale.shape[0],
        eps=attributes["epsilon"],
        initialize_parameters=False,
        requires_grad=context.requires_grad,
    )
    _load_state(module, weight=scale, bias=bias)
    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),))


# ONNX gate order and default activations, and the gate indices in PyTorch (and Warp-NN) order
_RECURRENT_LAYERS: dict[str, tuple[type[nn.Module], tuple[str, ...], tuple[int, ...]]] = {
    "RNN": (nn.RNN, ("tanh",), (0,)),  # (h)
    "GRU": (nn.GRU, ("sigmoid", "tanh"), (1, 0, 2)),  # (z, r, h) -> (r, z, n)
    "LSTM": (nn.LSTM, ("sigmoid", "tanh", "tanh"), (0, 2, 3, 1)),  # (i, o, f, c) -> (i, f, g, o)
}
_DIRECTIONS = {"forward": (False,), "reverse": (True,), "bidirectional": (False, True)}


def _compile_recurrent(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``RNN``/``GRU``/``LSTM`` node into a single-layer :class:`RNN`/:class:`GRU`/:class:`LSTM` module.

    Only the default activations are supported, and the GRU requires ``linear_before_reset = 1`` (as exported by
    PyTorch). The sequence lengths and the LSTM peepholes are not supported.
    """
    module_type, activations, gate_order = _RECURRENT_LAYERS[node.op_type]
    extra = {"GRU": {"linear_before_reset": 0}, "LSTM": {"input_forget": 0}}.get(node.op_type, {})
    attributes = _attributes(
        node,
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction="forward",
        hidden_size=None,
        layout=0,
        **extra,
    )
    if attributes["direction"] not in _DIRECTIONS:
        raise ValueError(f"OnnxRuntime {node.op_type}: invalid direction '{attributes['direction']}'")
    reverse = _DIRECTIONS[attributes["direction"]]
    unsupported = {
        "activation_alpha/activation_beta": attributes["activation_alpha"] or attributes["activation_beta"],
        "clip": attributes["clip"] is not None,
        "layout = 1": attributes["layout"],
        "non-default activations": attributes["activations"] is not None
        and tuple(a.lower() for a in attributes["activations"]) != activations * len(reverse),
        "linear_before_reset = 0": node.op_type == "GRU" and not attributes["linear_before_reset"],
        "input_forget": attributes.get("input_forget"),
        "sequence_lens": context.input_name(node, 4),
        "peepholes (P)": node.op_type == "LSTM" and context.input_name(node, 7),
    }
    if features := [name for name, used in unsupported.items() if used]:
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: {', '.join(features)} not supported")

    weight_ih, weight_hh = context.constant(node, 1), context.constant(node, 2)
    bias = context.constant(node, 3, required=False)
    hidden_size = weight_hh.shape[-1]
    if weight_ih.shape[0] != len(reverse):
        raise ValueError(f"OnnxRuntime {node.op_type}: W has {weight_ih.shape[0]} directions, expected {len(reverse)}")
    if attributes["hidden_size"] not in (None, hidden_size):
        raise ValueError(f"OnnxRuntime {node.op_type}: hidden_size {attributes['hidden_size']} does not match R")

    def reorder(array: np.ndarray) -> np.ndarray:
        return np.concatenate([array[i * hidden_size : (i + 1) * hidden_size] for i in gate_order])

    module = module_type(
        weight_ih.shape[-1],
        hidden_size,
        bidirectional=len(reverse) == 2,
        bias=bias is not None,
        initialize_parameters=False,
        requires_grad=context.requires_grad,
    )
    state = {}
    for d in range(len(reverse)):
        suffix = "_reverse" if d else ""
        state |= {f"weight_ih_l0{suffix}": reorder(weight_ih[d]), f"weight_hh_l0{suffix}": reorder(weight_hh[d])}
        if bias is not None:
            bias_ih, bias_hh = np.split(bias[d], 2)
            state |= {
                f"bias_ih_l0{suffix}": reorder(bias_ih)[:, None],
                f"bias_hh_l0{suffix}": reorder(bias_hh)[:, None],
            }
    _load_state(module, **state)

    initial_states = (context.tensor(node, 5, required=False),)
    if node.op_type == "LSTM":
        initial_states += (context.tensor(node, 6, required=False),)
    return _RecurrentOperation(
        node=node, module=module, flip=reverse == (True,), input=context.tensor(node, 0), initial_states=initial_states
    )


def _compile_identity(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Identity`` node into an :class:`Identity` module."""
    _attributes(node)
    return _ModuleOperation(
        node=node, module=nn.Identity(requires_grad=context.requires_grad), inputs=(context.tensor(node, 0),)
    )


def _compile_instance_norm(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``InstanceNormalization`` node into an :class:`InstanceNorm` module."""
    attributes = _attributes(node, epsilon=1e-5)
    scale, bias = context.constant(node, 1), context.constant(node, 2)
    module = nn.InstanceNorm(
        scale.shape[0],
        eps=attributes["epsilon"],
        affine=True,
        initialize_parameters=False,
        requires_grad=context.requires_grad,
    )
    _load_state(module, weight=scale, bias=bias)
    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),))


def _compile_trailing_norm(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``LayerNormalization``/``RMSNormalization`` node into a :class:`LayerNorm`/:class:`RMSNorm`.

    The normalized shape is given by the scale, whose shape must match the input dimensions from ``axis`` onward.
    """
    _check_outputs(node, 1)
    attributes = _attributes(node, axis=-1, epsilon=1e-5, stash_type=1)
    scale = context.constant(node, 1)
    if scale.ndim == 0:
        raise NotImplementedError(f"OnnxRuntime {node.op_type}: scalar scales are not supported")
    normalized_shape = scale.shape
    if node.op_type == "LayerNormalization":
        bias = context.constant(node, 2, required=False)
        module = nn.LayerNorm(
            normalized_shape,
            eps=attributes["epsilon"],
            bias=bias is not None,
            initialize_parameters=False,
            requires_grad=context.requires_grad,
        )
        _load_state(module, weight=scale, **({} if bias is None else {"bias": bias}))
    else:
        module = nn.RMSNorm(
            normalized_shape,
            eps=attributes["epsilon"],
            initialize_parameters=False,
            requires_grad=context.requires_grad,
        )
        _load_state(module, weight=scale)

    def forward(input: wp.array) -> wp.array:
        axis = attributes["axis"] % input.ndim if -input.ndim <= attributes["axis"] < input.ndim else None
        if axis is None or input.ndim - axis != len(normalized_shape):
            raise ValueError(
                f"OnnxRuntime {node.op_type}: the scale shape {normalized_shape} must match the input dimensions "
                f"from axis {attributes['axis']} onward, got input shape {tuple(input.shape)}"
            )
        return module(input)

    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),), forward=forward)


def _compile_softmax(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Softmax``/``LogSoftmax`` node into a :class:`Softmax`/:class:`LogSoftmax` module.

    Before opset 13, the input is coerced into 2D (the dimensions before and from ``axis`` onward).
    """
    module_type = nn.Softmax if node.op_type == "Softmax" else nn.LogSoftmax
    if context.opset >= 13:
        axis = _attributes(node, axis=-1)["axis"]
        return _ModuleOperation(
            node=node,
            module=module_type(dim=axis, requires_grad=context.requires_grad),
            inputs=(context.tensor(node, 0),),
        )
    axis = _attributes(node, axis=1)["axis"]
    module = module_type(dim=-1, requires_grad=context.requires_grad)

    def forward(input: wp.array) -> wp.array:
        shape = tuple(input.shape)
        start = axis + len(shape) if axis < 0 else axis
        view_shape = (math.prod(shape[:start]), math.prod(shape[start:]))
        return module(input.reshape(view_shape)).reshape(shape)

    return _ModuleOperation(node=node, module=module, inputs=(context.tensor(node, 0),), forward=forward)


def _compile_squeeze(node: _Node, context: CompilationContext) -> _Operation:
    """Compile an ONNX ``Squeeze`` node, whose axes are an attribute (opset < 13) or a constant input, into a view.

    If no axes are given, all the dimensions of size 1 are removed. An empty axes input removes no dimensions
    (as the ONNX reference implementation does).
    """
    if context.opset < 13:
        axes = _attributes(node, axes=None)["axes"]
    else:
        _attributes(node)
        axes = context.constant(node, 1, required=False)
    axes = None if axes is None else tuple(int(axis) for axis in np.ravel(axes))

    def output_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if axes is None:
            squeezed = {i for i, size in enumerate(shape) if size == 1}
        else:
            if any(not -len(shape) <= axis < len(shape) for axis in axes):
                raise ValueError(f"OnnxRuntime Squeeze: axes {axes} are out of range for input shape {shape}")
            squeezed = {axis % len(shape) for axis in axes}
            if len(squeezed) != len(axes):
                raise ValueError(f"OnnxRuntime Squeeze: axes {axes} contain duplicates for input shape {shape}")
            if any(shape[axis] != 1 for axis in squeezed):
                raise ValueError(f"OnnxRuntime Squeeze: cannot squeeze axes {axes} of input shape {shape}")
        output = tuple(size for i, size in enumerate(shape) if i not in squeezed)
        if not output:
            raise NotImplementedError(f"OnnxRuntime Squeeze: scalar (0D) outputs are not supported, got {shape}")
        return output

    return _ReshapeOperation(node=node, input=context.tensor(node, 0), output_shape=output_shape)


COMPILERS: dict[str, _Compiler] = {
    "Abs": _elementwise(nn.Abs),
    "Acos": _elementwise(nn.Acos),
    "Acosh": _elementwise(nn.Acosh),
    "Add": _elementwise(nn.Add, arity=2),
    "Asin": _elementwise(nn.Asin),
    "Asinh": _elementwise(nn.Asinh),
    "Atan": _elementwise(nn.Atan),
    "Atanh": _elementwise(nn.Atanh),
    "AveragePool": _compile_pool,
    "BatchNormalization": _compile_batch_norm,
    "BitwiseAnd": _elementwise(nn.BitwiseAnd, arity=2, differentiable=False),
    "BitwiseNot": _elementwise(nn.BitwiseNot, differentiable=False),
    "BitwiseOr": _elementwise(nn.BitwiseOr, arity=2, differentiable=False),
    "BitwiseXor": _elementwise(nn.BitwiseXor, arity=2, differentiable=False),
    "Ceil": _elementwise(nn.Ceil),
    "Celu": _elementwise(nn.CELU, attributes={"alpha": ("alpha", 1.0)}),
    "Clip": _compile_clip,
    "Conv": _compile_conv,
    "Cos": _elementwise(nn.Cos),
    "Cosh": _elementwise(nn.Cosh),
    "Div": _elementwise(nn.Div, arity=2),
    "Dropout": _compile_dropout,
    "Elu": _elementwise(nn.ELU, attributes={"alpha": ("alpha", 1.0)}),
    "Erf": _elementwise(nn.Erf),
    "Exp": _elementwise(nn.Exp),
    "Flatten": _compile_flatten,
    "Floor": _elementwise(nn.Floor),
    "Gelu": _elementwise(nn.GELU, attributes={"approximate": ("approximate", "none")}),
    "Gemm": _compile_linear,
    "GlobalAveragePool": _compile_global_pool,
    "GlobalMaxPool": _compile_global_pool,
    "GroupNormalization": _compile_group_norm,
    "GRU": _compile_recurrent,
    "HardSigmoid": _elementwise(nn.HardSigmoid, attributes={"alpha": ("alpha", 0.2), "beta": ("beta", 0.5)}),
    "HardSwish": _elementwise(nn.HardSwish),
    "Identity": _compile_identity,
    "InstanceNormalization": _compile_instance_norm,
    "LayerNormalization": _compile_trailing_norm,
    "LeakyRelu": _elementwise(nn.LeakyReLU, attributes={"negative_slope": ("alpha", 0.01)}),
    "Log": _elementwise(nn.Log),
    "LogSoftmax": _compile_softmax,
    "LSTM": _compile_recurrent,
    "MatMul": _compile_linear,
    "Max": _variadic(nn.Max),
    "MaxPool": _compile_pool,
    "Min": _variadic(nn.Min),
    "Mish": _elementwise(nn.Mish),
    "Mul": _elementwise(nn.Mul, arity=2),
    "Neg": _elementwise(nn.Neg),
    "Pow": _elementwise(nn.Pow, arity=2),
    "PRelu": _elementwise(nn.PReLU, arity=2),
    "Reciprocal": _elementwise(nn.Reciprocal),
    "Relu": _elementwise(nn.ReLU),
    "RMSNormalization": _compile_trailing_norm,
    "RNN": _compile_recurrent,
    "Round": _elementwise(nn.Round),
    "Selu": _elementwise(
        nn.SELU,
        attributes={"alpha": ("alpha", 1.67326319217681884765625), "scale": ("gamma", 1.05070102214813232421875)},
    ),
    "Shrink": _elementwise(nn.Shrink, attributes={"lambd": ("lambd", 0.5), "bias": ("bias", 0.0)}),
    "Sigmoid": _elementwise(nn.Sigmoid),
    "Sign": _elementwise(nn.Sign),
    "Sin": _elementwise(nn.Sin),
    "Sinh": _elementwise(nn.Sinh),
    "Softmax": _compile_softmax,
    "Softplus": _elementwise(nn.Softplus),
    "Softsign": _elementwise(nn.Softsign),
    "Sqrt": _elementwise(nn.Sqrt),
    "Squeeze": _compile_squeeze,
    "Sub": _elementwise(nn.Sub, arity=2),
    "Swish": _elementwise(nn.Swish, attributes={"alpha": ("alpha", 1.0)}),
    "Tan": _elementwise(nn.Tan),
    "Tanh": _elementwise(nn.Tanh),
    "ThresholdedRelu": _elementwise(nn.Threshold, attributes={"threshold": ("alpha", 1.0)}, value=0.0),
}
