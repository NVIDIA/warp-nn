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

"""Module-backed ONNX runtime.

The runtime parses an ONNX graph once and compiles each node into an operation executed by Warp-NN modules
(activations, operators and layers). Constant node inputs (initializers and ``Constant`` nodes), such as weights,
are loaded into the modules' parameters and buffers at compilation time.

Operations are shape-agnostic: symbolic ONNX dimensions are resolved from the actual input arrays, and each module
allocates (and caches) its output arrays on the first call with a given input shape. Subsequent calls with the same
input shapes are allocation-free and CUDA-graphable.
"""

from __future__ import annotations

from typing import Any

import warnings

from dataclasses import dataclass, field, replace

import numpy as np
import warp as wp

from warp_nn.runtime._onnx_compilers import COMPILERS, CompilationContext
from warp_nn.utils.device import parse_device


# ONNX parsing


def _require_onnx() -> Any:
    """Import the optional ONNX dependency when the runtime is constructed."""
    try:
        import onnx
        import onnx.helper
        import onnx.numpy_helper
    except ImportError as e:  # pragma: no cover - exercised only on missing dependency
        raise ImportError(
            "OnnxRuntime requires the optional `onnx` package. "
            "Install it with `pip install onnx>=1.16.0` or `pip install warp-nn[onnx]`."
        ) from e
    return onnx


_ATTRIBUTE_DECODERS = {
    1: lambda attribute: attribute.f,  # FLOAT
    2: lambda attribute: attribute.i,  # INT
    3: lambda attribute: attribute.s.decode("utf-8"),  # STRING
    4: lambda attribute: _require_onnx().numpy_helper.to_array(attribute.t),  # TENSOR
    6: lambda attribute: tuple(attribute.floats),  # FLOATS
    7: lambda attribute: tuple(attribute.ints),  # INTS
    8: lambda attribute: tuple(value.decode("utf-8") for value in attribute.strings),  # STRINGS
}


@dataclass(frozen=True, kw_only=True)
class _Node:
    """Device-independent representation of an ONNX graph node.

    Absent optional inputs and outputs are represented by empty names.
    """

    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_onnx(cls, node: Any) -> _Node:
        """Decode a standard-domain ONNX node into the internal representation.

        Attributes of unsupported types (e.g. graphs) are kept undecoded, so that operators reject them.
        """
        if node.domain not in ("", "ai.onnx"):
            raise NotImplementedError(
                f"OnnxRuntime: operator '{node.op_type}' from domain '{node.domain}' is not supported"
            )
        attributes = {}
        for attribute in node.attribute:
            decoder = _ATTRIBUTE_DECODERS.get(attribute.type)
            attributes[attribute.name] = attribute if decoder is None else decoder(attribute)
        return cls(
            op_type=node.op_type,
            inputs=tuple(node.input),
            outputs=tuple(node.output),
            attributes=attributes,
        )


@dataclass(frozen=True, kw_only=True)
class _InputSpec:
    """Shape and data type constraints declared by an ONNX graph input.

    Symbolic dimensions are represented by ``None``, and their (non-empty) names are collected in ``symbols``.
    """

    name: str
    dimensions: tuple[int | None, ...]
    dtype: type
    symbols: frozenset[str] = frozenset()

    @classmethod
    def from_onnx(cls, value: Any) -> _InputSpec:
        """Create an input specification, preserving symbolic dimensions."""
        spec = OnnxTensorSpec.from_onnx(value)
        symbols = frozenset(
            dimension.dim_param
            for dimension in value.type.tensor_type.shape.dim
            if dimension.HasField("dim_param") and dimension.dim_param
        )
        return cls(name=spec.name, dimensions=spec.shape, dtype=spec.dtype, symbols=symbols)

    def validate(self, array: wp.array) -> None:
        """Validate an actual input array against the ONNX declaration."""
        shape = tuple(array.shape)
        if len(shape) != len(self.dimensions):
            raise ValueError(
                f"OnnxRuntime: invalid input '{self.name}' rank. Expected {len(self.dimensions)}, got {len(shape)}"
            )
        for i, (actual, expected) in enumerate(zip(shape, self.dimensions)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"OnnxRuntime: invalid input '{self.name}' shape at dimension {i}. "
                    f"Expected {expected}, got {actual}"
                )
        if array.dtype != self.dtype:
            raise TypeError(
                f"OnnxRuntime: invalid input '{self.name}' dtype. "
                f"Expected {self.dtype.__name__}, got {array.dtype.__name__}"
            )
        if not array.is_contiguous:
            raise ValueError(f"OnnxRuntime: input '{self.name}' must be contiguous")


def _decode_constant_node(node: _Node) -> np.ndarray:
    """Decode the value of an ONNX ``Constant`` node."""
    if len(node.attributes) != 1:
        raise ValueError(f"OnnxRuntime Constant: exactly one value attribute is required, got {list(node.attributes)}")
    ((name, value),) = node.attributes.items()
    if name == "value":
        return np.asarray(value)
    if name in ("value_float", "value_floats"):
        return np.asarray(value, dtype=np.float32)
    if name in ("value_int", "value_ints"):
        return np.asarray(value, dtype=np.int64)
    raise NotImplementedError(f"OnnxRuntime Constant: attribute '{name}' is not supported")


# Runtime


@dataclass(frozen=True, kw_only=True)
class OnnxTensorSpec:
    """Name, shape and data type of an ONNX graph input or output.

    Symbolic (and otherwise unknown) dimensions are represented by ``None``.
    """

    name: str
    shape: tuple[int | None, ...]
    dtype: type

    @classmethod
    def from_onnx(cls, value: Any) -> OnnxTensorSpec:
        """Create a tensor specification, preserving symbolic dimensions."""
        tensor_type = value.type.tensor_type
        try:
            dtype = wp.dtype_from_numpy(_require_onnx().helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
        except (KeyError, TypeError) as e:
            raise NotImplementedError(
                f"OnnxRuntime: tensor '{value.name}' has an unsupported element type ({tensor_type.elem_type})"
            ) from e
        shape = tuple(
            dimension.dim_value if dimension.HasField("dim_value") and dimension.dim_value > 0 else None
            for dimension in tensor_type.shape.dim
        )
        return cls(name=value.name, shape=shape, dtype=dtype)


class OnnxRuntime:
    def __init__(
        self, path: str, *, device: str | wp.Device | None = None, requires_grad: bool = False, **kwargs: Any
    ) -> None:
        """Execute an ONNX graph using Warp-NN modules.

        Each graph node is compiled into Warp-NN modules (e.g. ``Gemm`` into :py:class:`~warp_nn.nn.Linear`,
        ``LSTM`` into :py:class:`~warp_nn.nn.LSTM`), with its constant inputs (initializers and ``Constant``
        nodes) loaded into the modules' parameters and buffers.

        Symbolic ONNX dimensions are resolved from the actual input arrays. The first call with a given set of
        input shapes allocates the (cached) output arrays of the modules and loads their kernels; subsequent calls
        with the same input shapes are allocation-free and can be captured in a CUDA graph
        (see :py:meth:`prepare`). The returned arrays are owned by the runtime and overwritten by subsequent calls
        (or, for pass-through nodes such as ``Identity``, are the input arrays themselves), so they must not be
        written to.

        :param path: Path to an ONNX model.
        :param device: Target Warp device. ``None`` selects the current default device.
        :param requires_grad: Whether the module parameters and cached output arrays require gradients.

        :raises NotImplementedError: If the model uses unsupported operators, attributes or inputs.
        :raises ValueError: If the model is invalid.

        Example:

            >>> import warp as wp
            >>> from warp_nn.runtime import OnnxRuntime
            >>>
            >>> # load the ONNX model
            >>> runtime = OnnxRuntime("policy.onnx")
            >>>
            >>> # run the graph with a batch of 4 observations
            >>> outputs = runtime({"obs": wp.zeros((4, 6), dtype=wp.float32)})
            >>> outputs["actions"].shape
            (4, 3)
        """
        # parse deprecated arguments
        # - batch size
        batch_size = kwargs.pop("batch_size", None)
        if batch_size is not None:
            warnings.warn(
                "The OnnxRuntime's 'batch_size' argument is deprecated and will be removed in a future release. "
                "Use OnnxRuntime.prepare(batch_size=...) method instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        # - input batch axes
        input_batch_axes = kwargs.pop("input_batch_axes", None)
        if input_batch_axes is not None:
            warnings.warn(
                "The OnnxRuntime's 'input_batch_axes' argument is deprecated and will be removed in a future release. "
                "Use OnnxRuntime.prepare(...) method instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if kwargs:
            raise TypeError(f"OnnxRuntime: unexpected keyword arguments {sorted(kwargs)}")

        self._device = parse_device(device)
        self._requires_grad = requires_grad
        self._prepared_shapes: dict[str, tuple[int, ...]] | None = None  # deprecated '_shapes' property

        # load and validate model
        onnx = _require_onnx()
        model = onnx.load(path)
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            raise ValueError(f"OnnxRuntime: invalid ONNX model: {e}") from e

        # collect graph metadata
        graph = model.graph
        constants = {
            initializer.name: np.asarray(onnx.numpy_helper.to_array(initializer)) for initializer in graph.initializer
        }
        self._input_specs = tuple(_InputSpec.from_onnx(value) for value in graph.input if value.name not in constants)
        self._inputs = tuple(
            OnnxTensorSpec(name=spec.name, shape=spec.dimensions, dtype=spec.dtype) for spec in self._input_specs
        )
        self._outputs = tuple(OnnxTensorSpec.from_onnx(value) for value in graph.output)

        # compile operations, creating the modules on the target device
        opset = next((entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")), 1)
        context = CompilationContext(
            device=self._device,
            requires_grad=requires_grad,
            opset=opset,
            constants=constants,
        )
        operations = []
        with wp.ScopedDevice(self._device):
            for onnx_node in graph.node:
                node = _Node.from_onnx(onnx_node)
                if node.op_type == "Constant":
                    constants[node.outputs[0]] = _decode_constant_node(node)
                    continue
                compiler = COMPILERS.get(node.op_type)
                if compiler is None:
                    supported = ", ".join(sorted(COMPILERS))
                    raise NotImplementedError(
                        f"OnnxRuntime: unsupported op '{node.op_type}'. Supported ops: {supported}"
                    )
                operations.append(compiler(node, context))
        self._operations = tuple(operations)

        # upload the constants read at runtime (including the constant graph outputs)
        self._constant_tensors = {}
        output_names = {spec.name for spec in self._outputs}
        for name in sorted(context.runtime_constants | (output_names & set(constants))):
            value = constants[name]
            if value.ndim == 0:
                raise NotImplementedError(f"OnnxRuntime: scalar constant '{name}' cannot be used as a runtime array")
            self._constant_tensors[name] = wp.array(
                np.ascontiguousarray(value), dtype=wp.dtype_from_numpy(value.dtype), device=self._device
            )

        # process deprecated arguments
        if batch_size is not None or input_batch_axes is not None:
            self._prepare_deprecated(1 if batch_size is None else batch_size, input_batch_axes)

    @property
    def inputs(self) -> tuple[OnnxTensorSpec, ...]:
        """Name, shape and data type of the graph inputs (excluding initializers), in declaration order.

        Example:

            >>> runtime.inputs
            (OnnxTensorSpec(name='obs', shape=(None, 6), dtype=<class '...float32'>),)
        """
        return self._inputs

    @property
    def outputs(self) -> tuple[OnnxTensorSpec, ...]:
        """Name, shape and data type of the graph outputs, in declaration order.

        Example:

            >>> runtime.outputs
            (OnnxTensorSpec(name='actions', shape=(None, 3), dtype=<class '...float32'>),)
        """
        return self._outputs

    @property
    def input_names(self) -> list[str]:
        """Names of the graph inputs (excluding initializers), in declaration order.

        .. deprecated:: 0.4.0

            Use :py:attr:`inputs` instead.
        """
        warnings.warn(
            "The OnnxRuntime's 'input_names' property is deprecated and will be removed in a future release. "
            "Use OnnxRuntime.inputs property instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [spec.name for spec in self._inputs]

    @property
    def output_names(self) -> list[str]:
        """Names of the graph outputs, in declaration order.

        .. deprecated:: 0.4.0

            Use :py:attr:`outputs` instead.
        """
        warnings.warn(
            "The OnnxRuntime's 'output_names' property is deprecated and will be removed in a future release. "
            "Use OnnxRuntime.outputs property instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [spec.name for spec in self._outputs]

    @property
    def _shapes(self) -> dict[str, tuple[int, ...]]:
        """Shapes of the graph inputs and outputs, keyed by name.

        The shapes are those of the last :py:meth:`prepare` call, or the declared ones (with the symbolic dimensions
        set to 1) if the runtime has not been prepared.

        .. deprecated:: 0.4.0

            Use :py:attr:`inputs`/:py:attr:`outputs`, or the shapes of the arrays returned by the runtime, instead.
        """
        warnings.warn(
            "The OnnxRuntime's '_shapes' property is deprecated and will be removed in a future release. "
            "Use OnnxRuntime.inputs/OnnxRuntime.outputs properties or the shapes of the output arrays instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._prepared_shapes is not None:
            return dict(self._prepared_shapes)
        return {
            spec.name: tuple(1 if dimension is None else dimension for dimension in spec.shape)
            for spec in (*self._inputs, *self._outputs)
        }

    def _prepare_deprecated(self, batch_size: int, input_batch_axes: int | dict[str, int] | None) -> None:
        """Prepare the runtime as the deprecated ``batch_size`` and ``input_batch_axes`` constructor arguments did.

        The batch axis of the inputs (all of them if ``input_batch_axes`` is an integer, or the given ones if it is
        a dictionary) is set to ``batch_size``, even if the ONNX model declares it as fixed, in which case the
        declaration is relaxed to accept any size (for subsequent :py:meth:`prepare` and :py:meth:`__call__` calls
        too). Every other symbolic dimension is also set to ``batch_size``.

        :raises KeyError: If ``input_batch_axes`` references unknown graph inputs.
        :raises ValueError: If ``batch_size`` is not positive, or if a batch axis is out of range.
        """
        if batch_size <= 0:
            raise ValueError(f"OnnxRuntime: 'batch_size' must be positive, got {batch_size}")
        if isinstance(input_batch_axes, dict):
            if unknown := set(input_batch_axes) - {spec.name for spec in self._input_specs}:
                raise KeyError(f"OnnxRuntime: 'input_batch_axes' references unknown graph inputs {sorted(unknown)}")
        input_specs, inputs = [], {}
        for spec in self._input_specs:
            axis = input_batch_axes.get(spec.name) if isinstance(input_batch_axes, dict) else input_batch_axes
            dimensions = list(spec.dimensions)
            if axis is not None:
                if not -len(dimensions) <= axis < len(dimensions):
                    raise ValueError(
                        f"OnnxRuntime: input '{spec.name}' batch axis {axis} is out of range "
                        f"for rank-{len(dimensions)} input"
                    )
                dimensions[axis] = None
            spec = replace(spec, dimensions=tuple(dimensions))
            input_specs.append(spec)
            # ones (rather than zeros) keep graphs that divide by integer inputs well defined
            inputs[spec.name] = wp.ones(
                tuple(batch_size if dimension is None else dimension for dimension in spec.dimensions),
                dtype=spec.dtype,
                device=self._device,
            )
        self._input_specs = tuple(input_specs)
        self._inputs = tuple(
            OnnxTensorSpec(name=spec.name, shape=spec.dimensions, dtype=spec.dtype) for spec in self._input_specs
        )
        self.prepare(inputs)

    def _validate_inputs(self, inputs: dict[str, wp.array]) -> None:
        """Validate the names, shapes, data types, devices and memory layout of the graph inputs."""
        expected_names = {spec.name for spec in self._inputs}
        provided_names = set(inputs)
        if missing := expected_names - provided_names:
            raise KeyError(f"OnnxRuntime: missing inputs {sorted(missing)}")
        if unknown := provided_names - expected_names:
            raise KeyError(f"OnnxRuntime: unknown inputs {sorted(unknown)}")
        for spec in self._input_specs:
            value = inputs[spec.name]
            spec.validate(value)
            if value.device != self._device:
                raise ValueError(
                    f"OnnxRuntime: input '{spec.name}' is on device {value.device}, expected {self._device}"
                )

    def prepare(self, inputs: dict[str, wp.array] | None = None, *, batch_size: int | None = None) -> None:
        """Run the graph once to allocate the cached arrays and load the kernels for the given input shapes.

        This must be done before capturing the runtime calls in a CUDA graph, since neither memory allocations
        nor kernel loading can be captured.

        Exactly one of ``inputs`` or ``batch_size`` must be given. When ``batch_size`` is given, one-filled inputs
        are generated from the ONNX declarations, with the symbolic (or unknown) batch dimension set to
        ``batch_size``. Each input must then declare at most one symbolic dimension, and all the named symbolic
        dimensions must share the same name; otherwise, the inputs must be given explicitly. Note that the symbolic
        dimension is assumed to be the batch one, so models with a fixed batch dimension and another symbolic
        dimension (e.g. a sequence length) must also be prepared with explicit inputs.

        :param inputs: Input arrays, keyed by graph input name.
        :param batch_size: Size of the batch dimension.

        :raises ValueError: If both or neither of ``inputs`` and ``batch_size`` are given, if ``batch_size``
            is not positive, or if the batch dimension is ambiguous.

        Example:

            >>> from warp_nn.utils import ScopedCapture
            >>>
            >>> # prepare the runtime to allow capturing the calls in a CUDA graph
            >>> runtime.prepare(batch_size=4)  # equivalent to runtime.prepare({"obs": obs})
            >>>
            >>> obs = wp.zeros((4, 6), dtype=wp.float32)
            >>> with ScopedCapture(enabled=wp.get_device().is_cuda) as capture:  # capture on CUDA devices only
            ...     actions = runtime({"obs": obs})["actions"]
            >>>
            >>> # launch the captured graph or run the runtime normally if no graph was captured
            >>> obs.fill_(1.0)  # update the inputs in place
            >>> if capture.graph is not None:
            ...     wp.capture_launch(capture.graph)
            ... else:
            ...     actions = runtime({"obs": obs})["actions"]
            >>> actions.shape
            (4, 3)
        """
        if (inputs is None) == (batch_size is None):
            raise ValueError("OnnxRuntime: exactly one of 'inputs' or 'batch_size' must be given")
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError(f"OnnxRuntime: 'batch_size' must be positive, got {batch_size}")
            for spec in self._input_specs:
                if spec.dimensions.count(None) > 1:
                    raise ValueError(
                        f"OnnxRuntime: input '{spec.name}' has more than one symbolic dimension, "
                        "so the batch dimension is ambiguous. Prepare the runtime with explicit inputs instead"
                    )
            symbols = set().union(*(spec.symbols for spec in self._input_specs))
            if len(symbols) > 1:
                raise ValueError(
                    f"OnnxRuntime: inputs have differently named symbolic dimensions {sorted(symbols)}, "
                    "so the batch dimension is ambiguous. Prepare the runtime with explicit inputs instead"
                )
            # ones (rather than zeros) keep graphs that divide by integer inputs well defined
            inputs = {
                spec.name: wp.ones(
                    tuple(batch_size if dimension is None else dimension for dimension in spec.dimensions),
                    dtype=spec.dtype,
                    device=self._device,
                )
                for spec in self._input_specs
            }
        outputs = self(inputs)
        # deprecated '_shapes' property
        self._prepared_shapes = {name: tuple(array.shape) for name, array in {**inputs, **outputs}.items()}

    def __call__(self, inputs: dict[str, wp.array]) -> dict[str, wp.array]:
        """Execute the graph.

        :param inputs: Input arrays, keyed by graph input name. They must be contiguous.

        :return: Output arrays, keyed by graph output name.

        Example:

            >>> obs = wp.zeros((4, 6), dtype=wp.float32)
            >>> outputs = runtime({"obs": obs})
            >>> list(outputs)
            ['actions']
            >>> runtime({"obs": obs})["actions"].ptr == outputs["actions"].ptr  # the outputs are cached per input shape
            True
        """
        self._validate_inputs(inputs)
        tensors = {**self._constant_tensors, **inputs}
        for operation in self._operations:
            operation(tensors)
        return {spec.name: tensors[spec.name] for spec in self._outputs}
