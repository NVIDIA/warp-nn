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

import pytest


onnx = pytest.importorskip("onnx")
import inspect
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from onnx import TensorProto, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import numpy as np
import warp as wp

from tests.utilities import is_device_available
from warp_nn import nn
from warp_nn.modules import activations, layers, operators
from warp_nn.runtime import OnnxRuntime, OnnxTensorSpec


_OPSET = 24
_RNG = np.random.default_rng(0)


def _uniform(shape, low=-3.0, high=3.0, dtype=np.float32):
    return _RNG.uniform(low, high, size=shape).astype(dtype)


def _integers(shape, dtype=np.int32):
    return _RNG.integers(-100, 100, size=shape).astype(dtype)


def _make_model(nodes, inputs, outputs, constants=None, opset=_OPSET, output_shapes=None) -> onnx.ModelProto:
    """Create a model from nodes, with graph inputs/outputs given as {name: numpy dtype} and constant initializers.

    The output shapes (required by the ONNX checker) are computed by running the ONNX reference implementation,
    unless they are given.
    """

    def make(output_shapes):
        def value_info(name, dtype, shape):
            return helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), shape)

        graph = helper.make_graph(
            nodes=nodes,
            name="graph",
            inputs=[value_info(name, value.dtype, value.shape) for name, value in inputs.items()],
            outputs=[value_info(name, dtype, shape) for (name, dtype), shape in zip(outputs.items(), output_shapes)],
            initializer=[numpy_helper.from_array(value, name) for name, value in (constants or {}).items()],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    if output_shapes is None:
        output_shapes = [output.shape for output in ReferenceEvaluator(make([None] * len(outputs))).run(None, inputs)]
    return make(output_shapes)


def _load(tmp_path, model, device, **kwargs) -> OnnxRuntime:
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return OnnxRuntime(str(path), device=device, **kwargs)


def _skip_if_unavailable(device, opset=_OPSET):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    if opset > defs.onnx_opset_version():
        pytest.skip(f"Opset {opset} is not supported by the installed onnx package")


def _module_types(runtime):
    return [type(module) for operation in runtime._operations for module in operation.modules]


# Single-node operator cases, checked against the ONNX reference implementation


@dataclass
class _Case:
    op_type: str
    modules: tuple[type, ...]
    inputs: dict[str, np.ndarray]
    constants: dict[str, np.ndarray] = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    node_inputs: tuple[str, ...] | None = None  # defaults to the graph inputs followed by the constants
    outputs: tuple[str, ...] = ("output",)
    opset: int = _OPSET
    reference: Callable[..., np.ndarray] | None = None  # defaults to the ONNX reference implementation
    output_shapes: tuple[tuple[int, ...], ...] | None = None  # defaults to the reference output shapes

    def expected(self) -> list[np.ndarray]:
        if self.reference is None:
            return ReferenceEvaluator(self.model()).run(None, self.inputs)
        return [self.reference(*self.inputs.values())]

    def model(self) -> onnx.ModelProto:
        node_inputs = (*self.inputs, *self.constants) if self.node_inputs is None else self.node_inputs
        node = helper.make_node(self.op_type, list(node_inputs), list(self.outputs), **self.attributes)
        dtype = next(iter(self.inputs.values())).dtype
        outputs = {name: dtype for name in self.outputs if name}
        return _make_model([node], self.inputs, outputs, self.constants, self.opset, self.output_shapes)


def _unary(op_type, module, x=None, **attributes):
    return _Case(op_type, (module,), {"x": _uniform((2, 3, 4)) if x is None else x}, attributes=attributes)


def _binary(op_type, module, a=None, b=None):
    return _Case(
        op_type, (module,), {"a": _uniform((2, 3, 4)) if a is None else a, "b": _uniform((2, 3, 4)) if b is None else b}
    )


def _recurrent_constants(gates, directions, input_size, hidden_size):
    return {
        "W": _uniform((directions, gates * hidden_size, input_size), -0.5, 0.5),
        "R": _uniform((directions, gates * hidden_size, hidden_size), -0.5, 0.5),
        "B": _uniform((directions, 2 * gates * hidden_size), -0.5, 0.5),
    }


_CASES = {
    # activations
    "Celu": _unary("Celu", activations.CELU, alpha=0.5),
    "Elu": _unary("Elu", activations.ELU, alpha=0.5),
    "Gelu": _unary("Gelu", activations.GELU),
    "Gelu_tanh": _unary("Gelu", activations.GELU, approximate="tanh"),
    "HardSigmoid": _unary("HardSigmoid", activations.HardSigmoid),
    "HardSwish": _unary("HardSwish", activations.HardSwish),
    "LeakyRelu": _unary("LeakyRelu", activations.LeakyReLU, alpha=0.2),
    "LogSoftmax": _unary("LogSoftmax", activations.LogSoftmax, axis=1),
    "LogSoftmax_opset11": _Case(
        "LogSoftmax",
        (activations.LogSoftmax,),
        {"x": _uniform((2, 3, 4))},
        attributes={"axis": -2},
        opset=11,
        # the input is coerced into 2D (the ONNX reference implementation does not follow opset 11 semantics)
        reference=lambda x: (x.reshape(2, 12) - np.log(np.exp(x.reshape(2, 12)).sum(1, keepdims=True))).reshape(
            x.shape
        ),
    ),
    "Mish": _unary("Mish", activations.Mish),
    "PRelu": _Case("PRelu", (activations.PReLU,), {"x": _uniform((2, 3, 4))}, {"slope": _uniform((4,), 0.0, 1.0)}),
    "Relu": _unary("Relu", activations.ReLU),
    "Relu_4d": _unary("Relu", activations.ReLU, x=_uniform((2, 3, 4, 5))),
    "Selu": _unary("Selu", activations.SELU),
    "Shrink": _unary("Shrink", activations.Shrink, lambd=1.0, bias=0.5),
    "Sigmoid": _unary("Sigmoid", activations.Sigmoid),
    "Softmax": _unary("Softmax", activations.Softmax),
    "Softmax_opset11": _Case(
        "Softmax",
        (activations.Softmax,),
        {"x": _uniform((2, 3, 4))},
        attributes={"axis": 1},
        opset=11,
        # the input is coerced into 2D (the ONNX reference implementation does not follow opset 11 semantics)
        reference=lambda x: (np.exp(x.reshape(2, 12)) / np.exp(x.reshape(2, 12)).sum(1, keepdims=True)).reshape(
            x.shape
        ),
    ),
    "Softplus": _unary("Softplus", activations.Softplus),
    "Softsign": _unary("Softsign", activations.Softsign),
    "Swish": _unary("Swish", activations.Swish, alpha=0.5),
    "ThresholdedRelu": _unary("ThresholdedRelu", activations.Threshold, alpha=0.5),
    # unary operators
    "Abs": _unary("Abs", operators.Abs),
    "Acos": _unary("Acos", operators.Acos, x=_uniform((2, 3, 4), -0.9, 0.9)),
    "Acosh": _unary("Acosh", operators.Acosh, x=_uniform((2, 3, 4), 1.0, 4.0)),
    "Asin": _unary("Asin", operators.Asin, x=_uniform((2, 3, 4), -0.9, 0.9)),
    "Asinh": _unary("Asinh", operators.Asinh),
    "Atan": _unary("Atan", operators.Atan),
    "Atanh": _unary("Atanh", operators.Atanh, x=_uniform((2, 3, 4), -0.9, 0.9)),
    "BitwiseNot": _unary("BitwiseNot", operators.BitwiseNot, x=_integers((2, 3, 4))),
    "Ceil": _unary("Ceil", operators.Ceil),
    "Clip": _Case(
        "Clip",
        (operators.Clip,),
        {"x": _uniform((2, 3, 4))},
        {"min": np.array(-1.0, np.float32), "max": np.array(0.5, np.float32)},
    ),
    "Clip_max_only": _Case(
        "Clip",
        (operators.Clip,),
        {"x": _uniform((2, 3, 4))},
        {"max": np.array(0.5, np.float32)},
        node_inputs=("x", "", "max"),
    ),
    "Clip_opset6": _Case("Clip", (operators.Clip,), {"x": _uniform((2, 3, 4))}, attributes={"min": -1.0}, opset=6),
    "Cos": _unary("Cos", operators.Cos),
    "Cosh": _unary("Cosh", operators.Cosh),
    "Erf": _unary("Erf", operators.Erf),
    "Exp": _unary("Exp", operators.Exp),
    "Floor": _unary("Floor", operators.Floor),
    "Log": _unary("Log", operators.Log, x=_uniform((2, 3, 4), 0.5, 3.0)),
    "Neg": _unary("Neg", operators.Neg),
    "Reciprocal": _unary("Reciprocal", operators.Reciprocal, x=_uniform((2, 3, 4), 0.5, 3.0)),
    "Round": _unary("Round", operators.Round, x=np.arange(-2.5, 3.5, 0.25, dtype=np.float32).reshape(2, 3, 4)),
    "Sign": _unary("Sign", operators.Sign, x=np.arange(-1.0, 1.4, 0.1, dtype=np.float32).round(1).reshape(2, 3, 4)),
    "Sin": _unary("Sin", operators.Sin),
    "Sinh": _unary("Sinh", operators.Sinh),
    "Sqrt": _unary("Sqrt", operators.Sqrt, x=_uniform((2, 3, 4), 0.5, 3.0)),
    # shape operators (executed as views, without modules)
    "Squeeze": _Case("Squeeze", (), {"x": _uniform((2, 1, 3, 1))}, {"axes": np.array([1, -1], np.int64)}),
    "Squeeze_all": _Case("Squeeze", (), {"x": _uniform((1, 3, 1))}),
    "Squeeze_opset11": _Case("Squeeze", (), {"x": _uniform((1, 3, 1))}, attributes={"axes": [0]}, opset=11),
    "Tan": _unary("Tan", operators.Tan, x=_uniform((2, 3, 4), -1.0, 1.0)),
    "Tanh": _unary("Tanh", operators.Tanh),
    # binary operators
    "Add": _binary("Add", operators.Add),
    "Add_broadcast": _binary("Add", operators.Add, b=_uniform((3, 1))),
    "Add_broadcast_4d": _binary("Add", operators.Add, a=_uniform((2, 1, 4, 1)), b=_uniform((3, 1, 5))),
    "Add_constant": _Case("Add", (operators.Add,), {"x": _uniform((2, 3))}, {"bias": _uniform((3,))}),
    "BitwiseAnd": _binary("BitwiseAnd", operators.BitwiseAnd, a=_integers((2, 3, 4)), b=_integers((2, 3, 4))),
    "BitwiseOr": _binary("BitwiseOr", operators.BitwiseOr, a=_integers((2, 3, 4)), b=_integers((4,))),
    "BitwiseXor": _binary("BitwiseXor", operators.BitwiseXor, a=_integers((2, 3, 4)), b=_integers((2, 3, 4))),
    "Div": _binary("Div", operators.Div, b=_uniform((2, 3, 4), 0.5, 3.0)),
    "Max": _binary("Max", operators.Max),
    "Max_single_input": _Case("Max", (layers.Identity,), {"x": _uniform((2, 3))}),
    "Max_variadic": _Case(
        "Max",
        (operators.Max, operators.Max),
        {"a": _uniform((2, 3, 4)), "b": _uniform((3, 1))},
        {"c": _uniform((4,))},
    ),
    "Min": _binary("Min", operators.Min),
    "Min_variadic_4d": _Case(
        "Min",
        (operators.Min, operators.Min),
        {"a": _uniform((2, 3, 4, 5)), "b": _uniform((2, 3, 4, 5)), "c": _uniform((2, 3, 4, 5))},
    ),
    "Mul": _binary("Mul", operators.Mul),
    "Mul_scalar": _Case("Mul", (operators.Mul,), {"x": _uniform((2, 3))}, {"scale": np.array(2.5, np.float32)}),
    "Pow": _binary("Pow", operators.Pow, a=_uniform((2, 3, 4), 0.5, 3.0)),
    "Pow_integer_exponent": _Case("Pow", (operators.Pow,), {"x": _uniform((2, 3))}, {"y": np.array(3, np.int64)}),
    "Sub": _binary("Sub", operators.Sub),
    # linear layers
    "Gemm": _Case(
        "Gemm",
        (layers.Linear,),
        {"x": _uniform((5, 6))},
        {"weight": _uniform((4, 6)), "bias": _uniform((4,))},
        attributes={"alpha": 0.5, "beta": 2.0, "transB": 1},
    ),
    "Gemm_no_bias": _Case("Gemm", (layers.Linear,), {"x": _uniform((5, 6))}, {"weight": _uniform((6, 4))}),
    "MatMul": _Case("MatMul", (layers.Linear,), {"x": _uniform((2, 3, 6))}, {"weight": _uniform((6, 4))}),
    # convolution layers
    "Conv_1d": _Case(
        "Conv",
        (layers.Conv1D,),
        {"x": _uniform((2, 4, 10))},
        {"weight": _uniform((6, 2, 3)), "bias": _uniform((6,))},
        attributes={"group": 2, "pads": [1, 1], "strides": [2]},
    ),
    "Conv_2d": _Case(
        "Conv",
        (layers.Conv2D,),
        {"x": _uniform((2, 3, 8, 8))},
        {"weight": _uniform((4, 3, 3, 3))},
        attributes={"pads": [1, 1, 1, 1], "dilations": [2, 2]},
    ),
    # pooling layers
    "AveragePool_1d": _Case(
        "AveragePool",
        (layers.AvgPool1D,),
        {"x": _uniform((2, 3, 9))},
        attributes={"kernel_shape": [3], "strides": [2], "pads": [1, 1], "count_include_pad": 1},
    ),
    "AveragePool_2d": _Case(
        "AveragePool",
        (layers.AvgPool2D,),
        {"x": _uniform((2, 3, 7, 7))},
        attributes={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    ),
    "GlobalAveragePool": _Case("GlobalAveragePool", (layers.GlobalAvgPool,), {"x": _uniform((2, 3, 7))}),
    # the ONNX reference implementation drops the channel dimension of 3D inputs
    "GlobalMaxPool": _Case("GlobalMaxPool", (layers.GlobalMaxPool,), {"x": _uniform((2, 3, 4, 5))}),
    "MaxPool_1d": _Case("MaxPool", (layers.MaxPool1D,), {"x": _uniform((2, 3, 9))}, attributes={"kernel_shape": [2]}),
    "MaxPool_2d": _Case(
        "MaxPool",
        (layers.MaxPool2D,),
        {"x": _uniform((2, 3, 7, 7))},
        attributes={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": 1},
    ),
    # normalization layers
    "BatchNormalization": _Case(
        "BatchNormalization",
        (layers.BatchNorm,),
        {"x": _uniform((2, 3, 4, 5))},
        {
            "scale": _uniform((3,)),
            "bias": _uniform((3,)),
            "mean": _uniform((3,)),
            "var": _uniform((3,), 0.5, 2.0),
        },
        attributes={"epsilon": 1e-3},
    ),
    "GroupNormalization": _Case(
        "GroupNormalization",
        (layers.GroupNorm,),
        {"x": _uniform((2, 4, 5))},
        {"scale": _uniform((4,)), "bias": _uniform((4,))},
        attributes={"num_groups": 2},
    ),
    "InstanceNormalization": _Case(
        "InstanceNormalization",
        (layers.InstanceNorm,),
        {"x": _uniform((2, 3, 5))},
        {"scale": _uniform((3,)), "bias": _uniform((3,))},
    ),
    "LayerNormalization": _Case(
        "LayerNormalization", (layers.LayerNorm,), {"x": _uniform((2, 3, 4))}, {"scale": _uniform((4,))}
    ),
    "LayerNormalization_axis": _Case(
        "LayerNormalization",
        (layers.LayerNorm,),
        {"x": _uniform((2, 3, 4))},
        {"scale": _uniform((3, 4)), "bias": _uniform((3, 4))},
        attributes={"axis": 1},
    ),
    "RMSNormalization": _Case(
        "RMSNormalization", (layers.RMSNorm,), {"x": _uniform((2, 3, 4))}, {"scale": _uniform((4,))}
    ),
    # shape and regularization layers
    "Dropout": _Case("Dropout", (layers.Dropout,), {"x": _uniform((2, 3))}, {"ratio": np.array(0.3, np.float32)}),
    "Dropout_opset10": _Case(
        "Dropout", (layers.Dropout,), {"x": _uniform((2, 3))}, attributes={"ratio": 0.3}, opset=10
    ),
    "Flatten": _Case("Flatten", (layers.Flatten,), {"x": _uniform((2, 3, 4, 5))}, attributes={"axis": 2}),
    "Flatten_axis0": _Case("Flatten", (layers.Flatten,), {"x": _uniform((2, 3, 4))}, attributes={"axis": 0}),
    "Identity": _Case("Identity", (layers.Identity,), {"x": _uniform((2, 3))}),
    # recurrent layers
    "RNN": _Case(
        "RNN",
        (layers.RNN,),
        {"x": _uniform((3, 2, 5)), "initial_h": _uniform((1, 2, 4))},
        _recurrent_constants(1, 1, 5, 4),
        attributes={"hidden_size": 4},
        node_inputs=("x", "W", "R", "B", "", "initial_h"),
        outputs=("y", "y_h"),
    ),
    "GRU_bidirectional": _Case(
        "GRU",
        (layers.GRU,),
        {"x": _uniform((3, 2, 5))},
        _recurrent_constants(3, 2, 5, 4),
        attributes={"hidden_size": 4, "direction": "bidirectional", "linear_before_reset": 1},
        outputs=("y", "y_h"),
    ),
    "LSTM_reverse": _Case(
        "LSTM",
        (layers.LSTM,),
        {"x": _uniform((3, 2, 5)), "initial_h": _uniform((1, 2, 4)), "initial_c": _uniform((1, 2, 4))},
        _recurrent_constants(4, 1, 5, 4),
        attributes={"hidden_size": 4, "direction": "reverse"},
        node_inputs=("x", "W", "R", "B", "", "initial_h", "initial_c"),
        outputs=("y", "y_h", "y_c"),
    ),
    "GRU_reverse": _Case(
        "GRU",
        (layers.GRU,),
        {"x": _uniform((3, 2, 5))},
        _recurrent_constants(3, 1, 5, 4),
        attributes={"hidden_size": 4, "direction": "reverse", "linear_before_reset": 1},
        outputs=("y", "y_h"),
    ),
    "LSTM_bidirectional": _Case(
        "LSTM",
        (layers.LSTM,),
        {"x": _uniform((3, 2, 5)), "initial_h": _uniform((2, 2, 4))},
        _recurrent_constants(4, 2, 5, 4),
        attributes={"hidden_size": 4, "direction": "bidirectional"},
        node_inputs=("x", "W", "R", "B", "", "initial_h"),
        outputs=("y", "y_h", "y_c"),
    ),
    "LSTM_final_state_only": _Case(
        "LSTM",
        (layers.LSTM,),
        {"x": _uniform((3, 2, 5))},
        {k: v for k, v in _recurrent_constants(4, 1, 5, 4).items() if k != "B"},
        attributes={"hidden_size": 4},
        node_inputs=("x", "W", "R"),
        outputs=("", "y_h"),
    ),
}


def test_all_modules_are_covered():
    """Every activation, operator and layer module (except containers and lazy variants) is used by some case."""
    excluded = {layers.LazyLinear, layers.Sequential, layers.RNNCell, layers.GRUCell, layers.LSTMCell}
    exported = {
        cls
        for package in (activations, operators, layers)
        for _, cls in inspect.getmembers(package, inspect.isclass)
        if issubclass(cls, nn.Module) and cls not in excluded
    }
    covered = {module for case in _CASES.values() for module in case.modules}
    assert exported - covered == set()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("case_id", list(_CASES))
def test_operators(tmp_path, device, case_id):
    case = _CASES[case_id]
    _skip_if_unavailable(device, case.opset)
    model = case.model()

    runtime = _load(tmp_path, model, device)
    assert _module_types(runtime) == list(case.modules)

    inputs = {name: wp.array(value, device=device) for name, value in case.inputs.items()}
    outputs = runtime(inputs)
    expected = case.expected()

    assert list(outputs) == [name for name in case.outputs if name]
    for output, reference in zip(outputs.values(), expected):
        assert output.shape == reference.shape
        np.testing.assert_allclose(output.numpy(), reference, rtol=1.0e-4, atol=1.0e-5)


# Graph-level behavior


def _mlp_model(batch="batch") -> onnx.ModelProto:
    constants = {
        "w1": _uniform((8, 6), -0.5, 0.5),
        "b1": _uniform((8,), -0.5, 0.5),
        "w2": _uniform((3, 8), -0.5, 0.5),
        "b2": _uniform((3,), -0.5, 0.5),
    }
    nodes = [
        helper.make_node("Gemm", ["obs", "w1", "b1"], ["h1"], transB=1),
        helper.make_node("Elu", ["h1"], ["a1"]),
        helper.make_node("Gemm", ["a1", "w2", "b2"], ["h2"], transB=1),
        helper.make_node("Tanh", ["h2"], ["actions"]),
    ]
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, (batch, 6))],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, (batch, 3))],
        initializer=[numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_symbolic_dimensions_and_cached_outputs(tmp_path, device):
    _skip_if_unavailable(device)
    model = _mlp_model()
    runtime = _load(tmp_path, model, device)
    assert _module_types(runtime) == [layers.Linear, activations.ELU, layers.Linear, operators.Tanh]
    # modules allocate their cached outputs lazily
    assert all(not module._cache for operation in runtime._operations for module in operation.modules)

    for batch in (1, 4):
        obs = _uniform((batch, 6))
        output = runtime({"obs": wp.array(obs, device=device)})["actions"]
        (expected,) = ReferenceEvaluator(model).run(None, {"obs": obs})
        np.testing.assert_allclose(output.numpy(), expected, rtol=1.0e-5, atol=1.0e-6)

    # the outputs are cached per input shape
    first = runtime({"obs": wp.zeros((4, 6), dtype=wp.float32, device=device)})["actions"]
    second = runtime({"obs": wp.ones((4, 6), dtype=wp.float32, device=device)})["actions"]
    assert first.ptr == second.ptr
    assert runtime({"obs": wp.zeros((1, 6), dtype=wp.float32, device=device)})["actions"].ptr != first.ptr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prepare(tmp_path, device):
    _skip_if_unavailable(device)
    runtime = _load(tmp_path, _mlp_model(), device)

    def cached_outputs():
        return {
            id(value) for operation in runtime._operations for m in operation.modules for value in m._cache.values()
        }

    # preparing from a batch size allocates the same cached outputs as preparing from inputs of that shape
    inputs = {"obs": wp.zeros((4, 6), dtype=wp.float32, device=device)}
    runtime.prepare(batch_size=4)
    prepared = cached_outputs()
    assert prepared
    runtime.prepare(inputs)
    assert cached_outputs() == prepared
    runtime.prepare(batch_size=2)
    assert cached_outputs() > prepared

    with pytest.raises(ValueError, match="exactly one of 'inputs' or 'batch_size'"):
        runtime.prepare()
    with pytest.raises(ValueError, match="exactly one of 'inputs' or 'batch_size'"):
        runtime.prepare(inputs, batch_size=4)
    for batch_size in (0, -1):
        with pytest.raises(ValueError, match=f"'batch_size' must be positive, got {batch_size}"):
            runtime.prepare(batch_size=batch_size)

    # multiple non-float inputs with fixed dimensions (the batch size does not apply)
    model = _make_model(
        [helper.make_node("BitwiseOr", ["a", "b"], ["output"])],
        {"a": _integers((2, 3)), "b": _integers((3,))},
        {"output": np.int32},
    )
    runtime = _load(tmp_path, model, device)
    runtime.prepare(batch_size=5)
    prepared = cached_outputs()
    runtime.prepare({name: wp.zeros(shape, dtype=wp.int32, device=device) for name, shape in (("a", (2, 3)), ("b", 3))})
    assert cached_outputs() == prepared

    # only the batch dimension is set, even if it is not the leading one
    def recurrent_model(sequence_length):
        graph = helper.make_graph(
            [helper.make_node("GRU", ["x", "W", "R", "B"], ["y", "y_h"], hidden_size=4, linear_before_reset=1)],
            "gru",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, (sequence_length, "batch", 5))],
            [helper.make_tensor_value_info("y_h", TensorProto.FLOAT, (1, "batch", 4))],
            initializer=[numpy_helper.from_array(v, k) for k, v in _recurrent_constants(3, 1, 5, 4).items()],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])

    runtime = _load(tmp_path, recurrent_model(2), device)
    runtime.prepare(batch_size=3)
    prepared = cached_outputs()
    runtime.prepare({"x": wp.zeros((2, 3, 5), dtype=wp.float32, device=device)})
    assert cached_outputs() == prepared

    with pytest.raises(ValueError, match="input 'x' has more than one symbolic dimension"):
        _load(tmp_path, recurrent_model("seq"), device).prepare(batch_size=3)

    # the batch dimension is shared by all the inputs, which are generated filled with ones
    def bitwise_model(a_batch, b_batch):
        graph = helper.make_graph(
            [helper.make_node("BitwiseAnd", ["a", "b"], ["output"])],
            "bitwise",
            [
                helper.make_tensor_value_info("a", TensorProto.INT32, (a_batch, 3)),
                helper.make_tensor_value_info("b", TensorProto.INT32, (b_batch, 3)),
            ],
            [helper.make_tensor_value_info("output", TensorProto.INT32, (a_batch, 3))],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])

    runtime = _load(tmp_path, bitwise_model("batch", "batch"), device)
    runtime.prepare(batch_size=4)
    ((output,),) = [module._cache.values() for operation in runtime._operations for module in operation.modules]
    np.testing.assert_array_equal(output.numpy(), np.ones((4, 3)))

    with pytest.raises(ValueError, match=r"differently named symbolic dimensions \['m', 'n'\]"):
        _load(tmp_path, bitwise_model("n", "m"), device).prepare(batch_size=4)

    # unnamed symbolic dimensions are compatible with the named one
    for a_batch, b_batch in (("", "batch"), (None, "batch")):
        runtime = _load(tmp_path, bitwise_model(a_batch, b_batch), device)
        runtime.prepare(batch_size=4)
        ((output,),) = [module._cache.values() for operation in runtime._operations for module in operation.modules]
        assert output.shape == (4, 3)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients(tmp_path, device):
    torch = pytest.importorskip("torch")
    _skip_if_unavailable(device)
    model = _mlp_model()
    runtime = _load(tmp_path, model, device, requires_grad=True)

    obs_np = _uniform((4, 6))
    obs = wp.array(obs_np, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        actions = runtime({"obs": obs})["actions"]
    upstream = _uniform(tuple(actions.shape), low=-1.0, high=1.0)  # non-uniform upstream gradients
    tape.backward(grads={actions: wp.array(upstream, device=device)})

    weights = {
        init.name: torch.tensor(numpy_helper.to_array(init), requires_grad=True) for init in model.graph.initializer
    }
    obs_torch = torch.tensor(obs_np, requires_grad=True)
    hidden = torch.nn.functional.elu(obs_torch @ weights["w1"].T + weights["b1"])
    (torch.tanh(hidden @ weights["w2"].T + weights["b2"]) * torch.tensor(upstream)).sum().backward()

    tolerances = {"rtol": 1.0e-4, "atol": 1.0e-5}
    np.testing.assert_allclose(obs.grad.numpy(), obs_torch.grad.numpy(), **tolerances)
    first_layer = runtime._operations[0].module
    np.testing.assert_allclose(first_layer.weight.data.grad.numpy(), weights["w1"].grad.numpy(), **tolerances)
    np.testing.assert_allclose(first_layer.bias.data.grad.numpy()[:, 0], weights["b1"].grad.numpy(), **tolerances)


_TORCH_GATE_ORDER = {"RNN": (0,), "GRU": (1, 0, 2), "LSTM": (0, 2, 3, 1)}  # ONNX gate indices in PyTorch order


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
@pytest.mark.parametrize("op_type", ["RNN", "GRU", "LSTM"])
@pytest.mark.parametrize("variant", ["full", "defaults", "partial_states"])  # (initial states, bias)
def test_recurrent_gradients(tmp_path, device, op_type, direction, variant):
    torch = pytest.importorskip("torch")
    _skip_if_unavailable(device)
    if variant == "partial_states" and op_type != "LSTM":
        pytest.skip("only the LSTM has more than one initial state")
    gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[op_type]
    directions = 2 if direction == "bidirectional" else 1
    input_size, hidden_size, seq_length, batch = 5, 4, 3, 2
    constants = _recurrent_constants(gates, directions, input_size, hidden_size)
    if variant == "defaults":
        del constants["B"]
    attributes = {"hidden_size": hidden_size, "direction": direction}
    if op_type == "GRU":
        attributes["linear_before_reset"] = 1
    states = {"full": ["initial_h", "initial_c"], "defaults": [], "partial_states": ["initial_h"]}[variant]
    states = [name for name in states if op_type == "LSTM" or name == "initial_h"]
    inputs = {"x": _uniform((seq_length, batch, input_size))} | {
        name: _uniform((directions, batch, hidden_size)) for name in states
    }
    node = helper.make_node(
        op_type,
        ["x", "W", "R", "B" if "B" in constants else "", "", *states],
        ["y", "y_h", *(["y_c"] if op_type == "LSTM" else [])],
        **attributes,
    )
    model = _make_model([node], inputs, {name: np.float32 for name in node.output}, constants)
    runtime = _load(tmp_path, model, device, requires_grad=True)

    arrays = {name: wp.array(value, device=device, requires_grad=True) for name, value in inputs.items()}
    tape = wp.Tape()
    with tape:
        outputs = runtime(arrays)
    # non-uniform upstream gradients, with the ONNX layouts: y (seq, directions, batch, hidden) and the final
    # states (directions, batch, hidden)
    upstream = {name: _uniform(tuple(array.shape), low=-1.0, high=1.0) for name, array in outputs.items()}
    tape.backward(grads={array: wp.array(upstream[name], device=device) for name, array in outputs.items()})

    # PyTorch reference (a reverse-only ONNX node is a unidirectional module applied to the time-reversed sequence)
    order = _TORCH_GATE_ORDER[op_type]

    def reorder(value):
        return np.concatenate([value[i * hidden_size : (i + 1) * hidden_size] for i in order])

    module = getattr(torch.nn, op_type)(
        input_size, hidden_size, bias="B" in constants, bidirectional=direction == "bidirectional"
    ).double()
    torch_inputs = {name: torch.tensor(value, dtype=torch.double, requires_grad=True) for name, value in inputs.items()}
    with torch.no_grad():
        for d in range(directions):
            suffix = "_reverse" if d else ""
            values = {
                f"weight_ih_l0{suffix}": reorder(constants["W"][d]),
                f"weight_hh_l0{suffix}": reorder(constants["R"][d]),
            }
            if "B" in constants:
                bias_ih, bias_hh = np.split(constants["B"][d], 2)
                values |= {f"bias_ih_l0{suffix}": reorder(bias_ih), f"bias_hh_l0{suffix}": reorder(bias_hh)}
            for name, value in values.items():
                getattr(module, name).copy_(torch.tensor(value))
    x = torch_inputs["x"]
    if direction == "reverse":
        # process the flipped sequence with a unidirectional module
        x = x.flip(0)
    hidden = tuple(
        torch_inputs[name] if name in states else torch.zeros(directions, batch, hidden_size, dtype=torch.double)
        for name in ("initial_h", "initial_c")[: 2 if op_type == "LSTM" else 1]
    )
    if not states:
        hidden = None
    elif op_type != "LSTM":
        hidden = hidden[0]
    y, final = module(x, hidden)
    if direction == "reverse":
        y = y.flip(0)
    # PyTorch's y has shape (seq, batch, directions * hidden), with the directions concatenated along the last axis
    y_upstream = torch.tensor(upstream["y"], dtype=torch.double).permute(0, 2, 1, 3).reshape(y.shape)
    final_states = final if op_type == "LSTM" else (final,)
    loss = (y * y_upstream).sum() + sum(
        (state * torch.tensor(upstream[name], dtype=torch.double)).sum()
        for name, state in zip(("y_h", "y_c"), final_states)
    )
    loss.backward()

    for name in ("x", *states):
        np.testing.assert_allclose(arrays[name].grad.numpy(), torch_inputs[name].grad.numpy(), rtol=1.0e-3, atol=1.0e-4)
    cell_module = runtime._operations[0].module
    for name, parameter in cell_module.named_parameters():
        expected = getattr(module, name).grad.numpy()
        np.testing.assert_allclose(
            parameter.data.grad.numpy().reshape(expected.shape), expected, rtol=1.0e-3, atol=1.0e-4
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recurrent_sequence_lengths(tmp_path, device):
    _skip_if_unavailable(device)
    constants = _recurrent_constants(3, 1, 5, 4)
    graph = helper.make_graph(
        [helper.make_node("GRU", ["x", "W", "R", "B"], ["y", "y_h"], hidden_size=4, linear_before_reset=1)],
        "gru",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ("seq", "batch", 5))],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, ("seq", 1, "batch", 4)),
            helper.make_tensor_value_info("y_h", TensorProto.FLOAT, (1, "batch", 4)),
        ],
        initializer=[numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    runtime = _load(tmp_path, model, device)

    for seq_length, batch in ((3, 2), (5, 2), (3, 2)):
        x = _uniform((seq_length, batch, 5))
        outputs = runtime({"x": wp.array(x, device=device)})
        for output, expected in zip(outputs.values(), ReferenceEvaluator(model).run(None, {"x": x})):
            np.testing.assert_allclose(output.numpy(), expected, rtol=1.0e-4, atol=1.0e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_constant_nodes(tmp_path, device):
    _skip_if_unavailable(device)
    bias = _uniform((3,))
    nodes = [
        helper.make_node("Constant", [], ["scale"], value_float=2.0),
        helper.make_node("Constant", [], ["bias"], value=numpy_helper.from_array(bias)),
        helper.make_node("Mul", ["x", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "bias"], ["output"]),
    ]
    model = _make_model(nodes, {"x": _uniform((2, 3))}, {"output": np.float32, "bias": np.float32})
    runtime = _load(tmp_path, model, device)
    assert _module_types(runtime) == [operators.Mul, operators.Add]

    x = _uniform((2, 3))
    outputs = runtime({"x": wp.array(x, device=device)})
    np.testing.assert_allclose(outputs["output"].numpy(), 2.0 * x + bias, rtol=1.0e-6)
    # constant graph outputs are uploaded as runtime arrays
    np.testing.assert_array_equal(outputs["bias"].numpy(), bias)


def test_cuda_graph_capture(tmp_path):
    _skip_if_unavailable("cuda")
    device = "cuda"
    constants = _recurrent_constants(4, 1, 5, 4) | {"offset": _uniform((2, 1))}
    nodes = [
        helper.make_node("LSTM", ["x", "W", "R", "B"], ["y", "y_h"], hidden_size=4),
        helper.make_node("Add", ["y_h", "offset"], ["shifted"]),
        helper.make_node("Add", ["shifted", "z"], ["output"]),  # runtime broadcast of z
    ]
    inputs = {"x": _uniform((3, 2, 5)), "z": _uniform((4,))}
    model = _make_model(nodes, inputs, {"output": np.float32}, constants)
    runtime = _load(tmp_path, model, device)

    arrays = {name: wp.array(value, device=device) for name, value in inputs.items()}
    runtime.prepare(arrays)
    with wp.ScopedCapture(device=device) as capture:
        output = runtime(arrays)["output"]

    for _ in range(2):
        new_inputs = {name: _uniform(value.shape) for name, value in inputs.items()}
        for name, value in new_inputs.items():
            arrays[name].assign(value)
        wp.capture_launch(capture.graph)
        (expected,) = ReferenceEvaluator(model).run(None, new_inputs)
        np.testing.assert_allclose(output.numpy(), expected, rtol=1.0e-4, atol=1.0e-5)


def test_inputs_and_outputs(tmp_path):
    runtime = _load(tmp_path, _mlp_model(), "cpu")
    assert runtime.inputs == (
        OnnxTensorSpec(name="obs", shape=(None, 6), dtype=wp.float32),
    )  # initializers are excluded
    assert runtime.outputs == (OnnxTensorSpec(name="actions", shape=(None, 3), dtype=wp.float32),)
    with pytest.raises(AttributeError):
        runtime.inputs = ()
    with pytest.raises(AttributeError):
        runtime.outputs = ()
    with pytest.raises(FrozenInstanceError):
        runtime.inputs[0].shape = (1, 6)


def test_deprecated_api(tmp_path):
    runtime = _load(tmp_path, _mlp_model(), "cpu")
    # input/output names
    with pytest.warns(DeprecationWarning, match="input_names"):
        assert runtime.input_names == ["obs"]
    with pytest.warns(DeprecationWarning, match="output_names"):
        assert runtime.output_names == ["actions"]
    # declared shapes (with symbolic dimensions set to 1) before preparing the runtime
    with pytest.warns(DeprecationWarning, match="_shapes"):
        assert runtime._shapes == {"obs": (1, 6), "actions": (1, 3)}
    # actual shapes of the last prepared inputs/outputs
    runtime.prepare(batch_size=4)
    with pytest.warns(DeprecationWarning, match="_shapes"):
        assert runtime._shapes == {"obs": (4, 6), "actions": (4, 3)}
    runtime.prepare({"obs": wp.zeros((2, 6), dtype=wp.float32, device="cpu")})
    with pytest.warns(DeprecationWarning, match="_shapes"):
        shapes = runtime._shapes
    assert shapes == {"obs": (2, 6), "actions": (2, 3)}
    shapes["actions"] = (1, 1)  # the returned dictionary is a copy
    with pytest.warns(DeprecationWarning, match="_shapes"):
        assert runtime._shapes["actions"] == (2, 3)
    # deprecated constructor arguments
    with pytest.warns(DeprecationWarning):
        runtime = _load(tmp_path, _mlp_model(), "cpu", batch_size=3, input_batch_axes=0)
    with pytest.warns(DeprecationWarning, match="_shapes"):
        assert runtime._shapes["actions"] == (3, 3)
    # deprecated batch axes overriding fixed (exported) dimensions
    x, h, c = _uniform((1, 1, 2)), _uniform((1, 1, 4)), _uniform((1, 1, 4))
    node = helper.make_node("LSTM", ["x", "W", "R", "B", "", "h", "c"], ["", "y_h", "y_c"], hidden_size=4)
    model = _make_model([node], {"x": x, "h": h, "c": c}, {"y_h": np.float32, "y_c": np.float32},
                        _recurrent_constants(4, 1, 2, 4))  # fmt: skip
    with pytest.warns(DeprecationWarning):
        runtime = _load(tmp_path, model, "cpu", batch_size=5, input_batch_axes={"x": 1, "h": -2, "c": 1})
    assert [spec.shape for spec in runtime.inputs] == [(1, None, 2), (1, None, 4), (1, None, 4)]
    with pytest.warns(DeprecationWarning, match="_shapes"):
        assert runtime._shapes == {"x": (1, 5, 2), "h": (1, 5, 4), "c": (1, 5, 4), "y_h": (1, 5, 4), "y_c": (1, 5, 4)}
    inputs = {name: np.repeat(value, 5, axis=1) for name, value in {"x": x, "h": h, "c": c}.items()}
    outputs = runtime({name: wp.array(value, device="cpu") for name, value in inputs.items()})
    expected = ReferenceEvaluator(model).run(None, inputs)
    for name, value in zip(("y_h", "y_c"), expected):
        np.testing.assert_allclose(outputs[name].numpy(), value, rtol=1e-5, atol=1e-6)
    outputs = runtime(
        {name: wp.array(np.repeat(value, 2, axis=1), device="cpu") for name, value in {"x": x, "h": h, "c": c}.items()}
    )
    assert outputs["y_h"].shape == (1, 2, 4)  # the relaxed batch axes accept any size
    with pytest.warns(DeprecationWarning):
        runtime = _load(tmp_path, model, "cpu", input_batch_axes={"x": 1, "h": None})  # batch size of 1
    assert [spec.shape for spec in runtime.inputs] == [(1, None, 2), (1, 1, 4), (1, 1, 4)]
    with pytest.warns(DeprecationWarning, match="_shapes"):
        assert runtime._shapes["y_h"] == (1, 1, 4)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="must be positive"):
        _load(tmp_path, model, "cpu", batch_size=0)
    with pytest.warns(DeprecationWarning), pytest.raises(KeyError, match="unknown graph inputs"):
        _load(tmp_path, model, "cpu", input_batch_axes={"y": 1})
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="out of range"):
        _load(tmp_path, model, "cpu", input_batch_axes=3)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="out of range"):
        _load(tmp_path, model, "cpu", input_batch_axes=-4)


def test_validates_inputs(tmp_path):
    runtime = _load(tmp_path, _mlp_model(), "cpu")

    with pytest.raises(KeyError, match="missing inputs"):
        runtime({})
    with pytest.raises(KeyError, match="unknown inputs"):
        runtime({"obs": wp.zeros((1, 6), dtype=wp.float32), "other": wp.zeros(1)})
    with pytest.raises(ValueError, match="invalid input 'obs' rank. Expected 2, got 1"):
        runtime({"obs": wp.zeros(6, dtype=wp.float32, device="cpu")})
    with pytest.raises(ValueError, match="invalid input 'obs' shape at dimension 1. Expected 6, got 5"):
        runtime({"obs": wp.zeros((1, 5), dtype=wp.float32, device="cpu")})
    with pytest.raises(TypeError, match="invalid input 'obs' dtype. Expected float32, got float64"):
        runtime({"obs": wp.zeros((1, 6), dtype=wp.float64, device="cpu")})
    with pytest.raises(ValueError, match="must be contiguous"):
        runtime({"obs": wp.zeros((6, 2), dtype=wp.float32, device="cpu").transpose()})
    if is_device_available("cuda"):
        with pytest.raises(ValueError, match="is on device cuda:0, expected cpu"):
            runtime({"obs": wp.zeros((1, 6), dtype=wp.float32, device="cuda")})


def test_rejects_non_broadcastable_operands(tmp_path):
    model = _make_model(
        [helper.make_node("Add", ["a", "b"], ["output"])],
        {"a": _uniform((2, 3)), "b": _uniform((2, 3))},
        {"output": np.float32},
    )
    # symbolic shapes are only validated at runtime
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "n"
    runtime = _load(tmp_path, model, "cpu")
    with pytest.raises(ValueError, match="not broadcastable"):
        runtime(
            {
                "a": wp.zeros((2, 4), dtype=wp.float32, device="cpu"),
                "b": wp.zeros((2, 3), dtype=wp.float32, device="cpu"),
            }
        )


_UNSUPPORTED = {
    "unsupported_op": (_Case("Hardmax", (), {"x": _uniform((2, 3))}), "unsupported op 'Hardmax'"),
    "unsupported_attribute": (
        _Case("Gemm", (), {"x": _uniform((6, 5))}, {"w": _uniform((6, 4))}, attributes={"transA": 1}),
        "transA is not supported",
    ),
    "dynamic_weight": (
        _Case("Gemm", (), {"x": _uniform((5, 6)), "w": _uniform((6, 4))}),
        "input 'w' must be a constant",
    ),
    "asymmetric_pads": (
        _Case(
            "Conv",
            (),
            {"x": _uniform((1, 2, 8))},
            {"w": _uniform((3, 2, 3))},
            attributes={"pads": [0, 1]},
        ),
        "only symmetric pads",
    ),
    "auto_pad_same": (
        _Case(
            "MaxPool",
            (),
            {"x": _uniform((1, 2, 8))},
            attributes={"kernel_shape": [3], "auto_pad": "SAME_UPPER"},
            output_shapes=((1, 2, 8),),  # the ONNX reference implementation fails for this model
        ),
        "auto_pad 'SAME_UPPER'",
    ),
    "maxpool_indices": (
        _Case(
            "MaxPool", (), {"x": _uniform((1, 2, 8))}, attributes={"kernel_shape": [2]}, outputs=("output", "indices")
        ),
        "only the first 1 output",
    ),
    "gru_linear_before_reset": (
        _Case("GRU", (), {"x": _uniform((3, 2, 5))}, _recurrent_constants(3, 1, 5, 4), attributes={"hidden_size": 4}),
        "linear_before_reset = 0",
    ),
    "sequence_lens": (
        _Case(
            "RNN",
            (),
            {"x": _uniform((3, 2, 5)), "lens": np.array([3, 2], np.int32)},
            _recurrent_constants(1, 1, 5, 4),
            attributes={"hidden_size": 4},
            node_inputs=("x", "W", "R", "B", "lens"),
        ),
        "sequence_lens",
    ),
    "rnn_activations": (
        _Case(
            "RNN",
            (),
            {"x": _uniform((3, 2, 5))},
            _recurrent_constants(1, 1, 5, 4),
            attributes={"hidden_size": 4, "activations": ["Relu"]},
            output_shapes=((3, 1, 2, 4),),  # the ONNX reference implementation does not support Relu
        ),
        "non-default activations",
    ),
    "lstm_attributes": (
        _Case(
            "LSTM",
            (),
            {"x": _uniform((3, 2, 5))},
            _recurrent_constants(4, 1, 5, 4),
            attributes={"hidden_size": 4, "clip": 1.0, "input_forget": 1},
        ),
        "clip, input_forget not supported",
    ),
    "lstm_layout": (
        _Case(
            "LSTM",
            (),
            {"x": _uniform((2, 3, 5))},
            _recurrent_constants(4, 1, 5, 4),
            attributes={"hidden_size": 4, "layout": 1},
        ),
        "layout = 1",
    ),
    "lstm_peepholes": (
        _Case(
            "LSTM",
            (),
            {"x": _uniform((3, 2, 5))},
            _recurrent_constants(4, 1, 5, 4) | {"P": _uniform((1, 12))},
            attributes={"hidden_size": 4},
            node_inputs=("x", "W", "R", "B", "", "", "", "P"),
        ),
        "peepholes",
    ),
    "gemm_bias_per_row": (
        _Case("Gemm", (), {"x": _uniform((5, 6))}, {"w": _uniform((6, 4)), "c": _uniform((5, 4))}),
        "C must be broadcastable",
    ),
    "batch_norm_training": (
        _Case(
            "BatchNormalization",
            (),
            {"x": _uniform((2, 3, 4))},
            {name: _uniform((3,), 0.5, 1.0) for name in ("scale", "bias", "mean", "var")},
            attributes={"training_mode": 1},
        ),
        "only spatial inference mode",
    ),
    "dropout_training": (
        _Case(
            "Dropout",
            (),
            {"x": _uniform((2, 3))},
            {"ratio": np.array(0.3, np.float32), "training": np.array(True)},
        ),
        "training mode is not supported",
    ),
    "average_pool_dilations": (
        _Case("AveragePool", (), {"x": _uniform((1, 2, 8))}, attributes={"kernel_shape": [2], "dilations": [2]}),
        "dilations are not supported",
    ),
    "conv_3d": (
        _Case("Conv", (), {"x": _uniform((1, 2, 4, 4, 4))}, {"w": _uniform((3, 2, 2, 2, 2))}),
        "only 1D and 2D kernels",
    ),
    "constant_string": (
        _Case(
            "Constant", (), {"x": _uniform((2,))}, attributes={"value_string": "a"}, node_inputs=(), output_shapes=((),)
        ),
        "attribute 'value_string' is not supported",
    ),
    "scalar_runtime_constant": (
        _Case("Identity", (), {"x": _uniform((2,))}, {"c": np.array(1.0, np.float32)}, node_inputs=("c",)),
        "scalar constant 'c'",
    ),
    "constant_only_operands": (
        _Case("Add", (), {"x": _uniform((2,))}, {"a": _uniform((2,)), "b": _uniform((2,))}, node_inputs=("a", "b")),
        "only constant inputs",
    ),
}


@pytest.mark.parametrize("case_id", list(_UNSUPPORTED))
def test_rejects_unsupported_models(tmp_path, case_id):
    case, message = _UNSUPPORTED[case_id]
    with pytest.raises(NotImplementedError, match=message):
        _load(tmp_path, case.model(), "cpu")


# models whose (static) input shapes do not match their weights, which is only detected when running them
_INVALID_SHAPES = {
    "squeeze_axes_out_of_range": (
        _Case("Squeeze", (), {"x": _uniform((2, 1))}, {"axes": np.array([3], np.int64)}, output_shapes=((2,),)),
        "axes \\(3,\\) are out of range for input shape \\(2, 1\\)",
    ),
    "squeeze_duplicate_axes": (
        _Case(
            "Squeeze", (), {"x": _uniform((2, 1, 1))}, {"axes": np.array([1, -2], np.int64)}, output_shapes=((2, 1),)
        ),
        "axes \\(1, -2\\) contain duplicates",
    ),
    "squeeze_axis": (
        _Case("Squeeze", (), {"x": _uniform((2, 3))}, {"axes": np.array([1], np.int64)}, output_shapes=((2,),)),
        "cannot squeeze axes \\(1,\\) of input shape \\(2, 3\\)",
    ),
    "linear_features": (
        _Case("Gemm", (), {"x": _uniform((5, 7))}, {"w": _uniform((6, 4))}, output_shapes=((5, 4),)),
        "expected 6 input features",
    ),
    "conv_channels": (
        _Case("Conv", (), {"x": _uniform((1, 3, 8))}, {"w": _uniform((4, 2, 3))}, output_shapes=((1, 4, 6),)),
        "expected an input with shape \\(batch_size, 2, \\*spatial\\)",
    ),
    "layer_norm_axis": (
        _Case(
            "LayerNormalization",
            (),
            {"x": _uniform((2, 3, 4))},
            {"scale": _uniform((4,))},
            attributes={"axis": 1},
            output_shapes=((2, 3, 4),),
        ),
        "the scale shape \\(4,\\) must match",
    ),
    "recurrent_features": (
        _Case(
            "RNN",
            (),
            {"x": _uniform((3, 2, 6))},
            _recurrent_constants(1, 1, 5, 4),
            attributes={"hidden_size": 4},
            outputs=("", "y_h"),
            output_shapes=((1, 2, 4),),
        ),
        "expected an input with shape \\(seq_length, batch_size, 5\\)",
    ),
    "recurrent_initial_state": (
        _Case(
            "RNN",
            (),
            {"x": _uniform((3, 2, 5)), "initial_h": _uniform((1, 3, 4))},
            _recurrent_constants(1, 1, 5, 4),
            attributes={"hidden_size": 4},
            node_inputs=("x", "W", "R", "B", "", "initial_h"),
            outputs=("", "y_h"),
            output_shapes=((1, 2, 4),),
        ),
        "initial state 'initial_h' must have shape \\(1, 2, 4\\)",
    ),
}


@pytest.mark.parametrize("case_id", list(_INVALID_SHAPES))
def test_rejects_invalid_shapes_at_runtime(tmp_path, case_id):
    case, message = _INVALID_SHAPES[case_id]
    runtime = _load(tmp_path, case.model(), "cpu")
    with pytest.raises(ValueError, match=message):
        runtime({name: wp.array(value, device="cpu") for name, value in case.inputs.items()})


def test_rejects_scalar_squeeze_outputs(tmp_path):
    case = _Case("Squeeze", (), {"x": _uniform((1,))}, output_shapes=((),))
    runtime = _load(tmp_path, case.model(), "cpu")
    with pytest.raises(NotImplementedError, match="scalar \\(0D\\) outputs are not supported"):
        runtime({"x": wp.zeros(1, dtype=wp.float32, device="cpu")})


def test_rejects_operands_with_different_dtypes(tmp_path):
    case = _Case("Pow", (), {"x": _uniform((2, 3)), "y": _integers((3,))}, output_shapes=((2, 3),))
    runtime = _load(tmp_path, case.model(), "cpu")
    with pytest.raises(NotImplementedError, match="different dtypes \\(float32, int32\\)"):
        runtime({name: wp.array(value, device="cpu") for name, value in case.inputs.items()})


def test_rejects_runtime_broadcasting_with_gradients(tmp_path):
    runtime = _load(tmp_path, _binary("Add", operators.Add, b=_uniform((4,))).model(), "cpu", requires_grad=True)
    with pytest.raises(NotImplementedError, match="broadcasting runtime operands is not supported with gradients"):
        runtime(
            {"a": wp.zeros((2, 3, 4), dtype=wp.float32, device="cpu"), "b": wp.zeros(4, dtype=wp.float32, device="cpu")}
        )


def test_rejects_custom_domains(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["output"], domain="com.example")],
        "graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (2,))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (2,))],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET), helper.make_opsetid("com.example", 1)]
    )
    with pytest.raises(NotImplementedError, match="from domain 'com.example'"):
        _load(tmp_path, model, "cpu")
