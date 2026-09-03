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

"""Graph-capturable ONNX inference runtime for Warp-NN policy networks.

Only the ``onnx`` package (pure protobuf parser) is required -- no
``onnxruntime`` or ``torch``.  Weights are loaded once onto the target
Warp device; inference executes a pre-built list of lightweight op
descriptors that dispatch to dedicated inference kernels without host
round-trips or device allocation.

Dense layers reuse Warp-NN's tiled matrix multiplication for efficient
single-policy and batched-policy execution. Elementwise and normalization
operators use deterministic one-writer kernels. All runtime-owned buffers
are allocated during construction, so execution is CUDA-graph capturable
after warmup.

Supported ONNX operators (all graph-capturable after one warmup call):

* **Add**, **Sub**, **Mul**, **Div** -- 2-D tensors with optional 1-D broadcasting
* **BatchNormalization** -- 2-D inference mode
* **Gemm** -- ``C = alpha * A @ B.T + beta * bias`` with ``transB=1``
* **Elu**, **Relu**, **Sqrt**, **Tanh** -- elementwise activation/math
* **ReduceMean** -- 2-D row reduction with ``keepdims=1``
* **Squeeze** -- alias passthrough (the output array shares memory with the
  input). Only used to drop unit dims, no copy is performed.
* **LSTM** -- forward, single-direction, single-layer, ``seq_length=1``. The
  full step (gate GEMM + cell update) executes in two on-device kernels.

Example::

    from warp_nn.runtime import OnnxRuntime

    rt = OnnxRuntime("policy.onnx", device="cuda:0")
    out = rt({"observation": wp.array2d(obs, dtype=wp.float32, device="cuda:0")})
    actions = out["action"]
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warp_nn.modules.layers._common import tile_transposed_gemm_2d
from warp_nn.utils.config import get_kernel_config
from warp_nn.utils.device import parse_device
from warp_nn.utils.ops import resolve_dim


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "OnnxRuntime requires the optional `onnx` package. "
            "Install it with `pip install onnx>=1.16.0` or `pip install warp-nn[onnx]`."
        ) from exc
    return onnx, numpy_helper


# ---------------------------------------------------------------------------
# Inference kernels
# ---------------------------------------------------------------------------
#
# Simple per-output-element kernels: one thread writes one cell.  Policies
# seen in practice are tiny (batch=1, hidden<=128), so the tiled variants
# used by the training modules are unnecessary here.


@wp.kernel
def _gemm_transb_kernel(
    A: wp.array2d[float],  # (M, K)
    B: wp.array2d[float],  # (N, K) — stored transposed
    bias: wp.array[float],  # (N,)
    C: wp.array2d[float],  # (M, N)
    K: int,
    alpha: float,
    beta: float,
):
    """``C = alpha * A @ B.T + beta * bias`` with ``transB=1``."""
    i, j = wp.tid()

    s = float(0.0)
    for k in range(K):
        s += A[i, k] * B[j, k]

    C[i, j] = alpha * s + beta * bias[j]


def _create_gemm_transb_tiled_kernel(config):
    @wp.kernel
    def kernel(
        A: wp.array2d[float],
        B: wp.array2d[float],
        bias: wp.array2d[float],
        alpha: float,
        beta: float,
        C: wp.array2d[float],
    ):
        i, j = wp.tid()
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        out = wp.static(tile_transposed_gemm_2d(config.tile_2d))(B, A, index=(i, j))
        shape_t = (wp.static(config.tile_2d[1]), wp.static(config.tile_2d[0]))
        shape_b = (wp.static(config.tile_2d[1]), 1)
        offset_b = (j * wp.static(config.tile_2d[1]), 0)
        tiled_bias = wp.tile_broadcast(wp.tile_load(bias, shape=shape_b, offset=offset_b), shape=shape_t)
        wp.tile_store(C, wp.tile_transpose(alpha * out + beta * tiled_bias), offset=offset)

    return kernel


_GEMM_CONFIG = get_kernel_config()
_GEMM_TRANSB_TILED_KERNEL = _create_gemm_transb_tiled_kernel(_GEMM_CONFIG)


@wp.kernel
def _elu_kernel(
    x: wp.array2d[float],
    y: wp.array2d[float],
    alpha: float,
):
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(v >= 0.0, v, alpha * (wp.exp(v) - 1.0))


@wp.kernel
def _unary_kernel(x: wp.array2d[float], operation: int, y: wp.array2d[float]):
    i, j = wp.tid()
    value = x[i, j]
    if operation == 0:
        y[i, j] = wp.max(value, 0.0)
    elif operation == 1:
        y[i, j] = wp.tanh(value)
    else:
        y[i, j] = wp.sqrt(value)


@wp.kernel
def _binary_broadcast_kernel(
    lhs: wp.array2d[float],
    rhs: wp.array2d[float],
    operation: int,
    out: wp.array2d[float],
):
    i, j = wp.tid()
    left = lhs[i % lhs.shape[0], j % lhs.shape[1]]
    right = rhs[i % rhs.shape[0], j % rhs.shape[1]]
    if operation == 0:
        out[i, j] = left + right
    elif operation == 1:
        out[i, j] = left - right
    elif operation == 2:
        out[i, j] = left * right
    else:
        out[i, j] = left / right


@wp.kernel
def _reduce_mean_rows_kernel(x: wp.array2d[float], out: wp.array2d[float]):
    row = wp.tid()
    total = float(0.0)
    for column in range(x.shape[1]):
        total += x[row, column]
    out[row, 0] = total / float(x.shape[1])


@wp.kernel
def _batch_normalization_kernel(
    x: wp.array2d[float],
    scale: wp.array[float],
    bias: wp.array[float],
    mean: wp.array[float],
    variance: wp.array[float],
    epsilon: float,
    relu: bool,
    y: wp.array2d[float],
):
    row, column = wp.tid()
    unit = (x[row, column] - mean[column]) / wp.sqrt(variance[column] + epsilon)
    value = unit * scale[column] + bias[column]
    y[row, column] = wp.where(relu, wp.max(value, 0.0), value)


@wp.func
def _inverse_sqrt(value: float):
    return 1.0 / wp.sqrt(value)


def _create_rms_normalization_kernel(width: int):
    """Create a deterministic one-block-per-row RMS normalization kernel."""

    @wp.kernel
    def kernel(
        x: wp.array2d[float],
        epsilon: wp.array[float],
        scale: wp.array[float],
        output: wp.array2d[float],
    ):
        row = wp.tid()
        values = wp.tile_load(x, shape=(1, wp.static(width)), offset=(row, 0))
        sum_squares = wp.tile_sum(values * values)
        epsilon_tile = wp.tile_load(epsilon, shape=(1,), offset=(0,))
        inverse_rms = wp.tile_map(_inverse_sqrt, sum_squares / float(wp.static(width)) + epsilon_tile)
        inverse_rms = wp.tile_broadcast(inverse_rms, shape=(1, wp.static(width)))
        scales = wp.tile_broadcast(
            wp.tile_load(scale, shape=(wp.static(width),), offset=(0,)),
            shape=(1, wp.static(width)),
        )
        wp.tile_store(output, values * inverse_rms * scales, offset=(row, 0))

    return kernel


@wp.kernel
def _lstm_gates_kernel(
    x: wp.array2d[float],  # (batch, input_size)
    h_prev: wp.array2d[float],  # (batch, hidden_size)
    W: wp.array2d[float],  # (4*hidden_size, input_size)
    R: wp.array2d[float],  # (4*hidden_size, hidden_size)
    gates: wp.array2d[float],  # (batch, 4*hidden_size) output
    input_size: int,
    hidden_size: int,
):
    """``gates = x @ W.T + h_prev @ R.T`` (one thread per (batch, gate))."""
    b, j = wp.tid()

    s = float(0.0)
    for k in range(input_size):
        s += x[b, k] * W[j, k]
    for k in range(hidden_size):
        s += h_prev[b, k] * R[j, k]

    gates[b, j] = s


@wp.kernel
def _lstm_cell_update_kernel(
    gates: wp.array2d[float],  # (batch, 4*hidden_size); already x@W.T + h_prev@R.T
    c_prev: wp.array2d[float],  # (batch, hidden_size)
    Bx: wp.array[float],  # (4*hidden_size,)
    Bh: wp.array[float],  # (4*hidden_size,)
    h_out: wp.array2d[float],  # (batch, hidden_size)
    c_out: wp.array2d[float],  # (batch, hidden_size)
    hidden_size: int,
):
    b, h = wp.tid()

    s_i = gates[b, 0 * hidden_size + h] + Bx[0 * hidden_size + h] + Bh[0 * hidden_size + h]
    s_o = gates[b, 1 * hidden_size + h] + Bx[1 * hidden_size + h] + Bh[1 * hidden_size + h]
    s_f = gates[b, 2 * hidden_size + h] + Bx[2 * hidden_size + h] + Bh[2 * hidden_size + h]
    s_c = gates[b, 3 * hidden_size + h] + Bx[3 * hidden_size + h] + Bh[3 * hidden_size + h]

    g_i = 1.0 / (1.0 + wp.exp(-s_i))
    g_o = 1.0 / (1.0 + wp.exp(-s_o))
    g_f = 1.0 / (1.0 + wp.exp(-s_f))
    g_c = wp.tanh(s_c)

    c_new = g_f * c_prev[b, h] + g_i * g_c
    c_out[b, h] = c_new
    h_out[b, h] = g_o * wp.tanh(c_new)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_ATTR_DECODERS = {
    1: lambda a: a.f,  # FLOAT
    2: lambda a: a.i,  # INT
    3: lambda a: a.s.decode("utf-8") if isinstance(a.s, (bytes, bytearray)) else a.s,  # STRING
    7: lambda a: list(a.ints),  # INTS
}


@dataclass
class _Op:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    attr_names: set[str] = field(default_factory=set)


def _decode_attrs(node) -> tuple[dict[str, Any], set[str]]:
    out: dict[str, Any] = {}
    all_names: set[str] = set()
    for attr in node.attribute:
        all_names.add(attr.name)
        decoder = _ATTR_DECODERS.get(attr.type)
        if decoder is not None:
            out[attr.name] = decoder(attr)
    return out, all_names


def _fuse_inference_ops(ops: list[_Op], graph_outputs: set[str], initializer_names: set[str]) -> list[_Op]:
    """Fuse common inference chains without changing the ONNX artifact."""
    consumers: dict[str, int] = {}
    for op in ops:
        for name in op.inputs:
            consumers[name] = consumers.get(name, 0) + 1
    fused: list[_Op] = []
    index = 0
    while index < len(ops):
        if index + 1 < len(ops):
            norm, relu = ops[index : index + 2]
            norm_out = norm.outputs[0]
            matches = (
                norm.op_type == "BatchNormalization"
                and len(norm.inputs) == 5
                and len(norm.outputs) == 1
                and int(norm.attrs.get("training_mode", 0)) == 0
                and relu.op_type == "Relu"
                and relu.inputs[0] == norm_out
                and consumers.get(norm_out) == 1
                and norm_out not in graph_outputs
            )
            if matches:
                fused.append(
                    _Op(
                        op_type="_BatchNormalizationRelu",
                        inputs=list(norm.inputs),
                        outputs=list(relu.outputs),
                        attrs={"epsilon": norm.attrs.get("epsilon", 1.0e-5)},
                    )
                )
                index += 2
                continue
        if index + 4 < len(ops):
            square, reduce, add, sqrt, divide = ops[index : index + 5]
            x_name = square.inputs[0] if len(square.inputs) == 2 and square.inputs[0] == square.inputs[1] else ""
            square_out = square.outputs[0]
            reduce_out = reduce.outputs[0]
            add_out = add.outputs[0]
            sqrt_out = sqrt.outputs[0]
            epsilon_name = ""
            if len(add.inputs) == 2:
                if add.inputs[0] == reduce_out:
                    epsilon_name = add.inputs[1]
                elif add.inputs[1] == reduce_out:
                    epsilon_name = add.inputs[0]
            matches = (
                square.op_type == "Mul"
                and bool(x_name)
                and reduce.op_type == "ReduceMean"
                and reduce.inputs[0] == square_out
                and tuple(int(axis) for axis in reduce.attrs.get("axes", [])) in ((1,), (-1,))
                and int(reduce.attrs.get("keepdims", 1)) == 1
                and add.op_type == "Add"
                and bool(epsilon_name)
                and sqrt.op_type == "Sqrt"
                and sqrt.inputs[0] == add_out
                and divide.op_type == "Div"
                and divide.inputs == [x_name, sqrt_out]
                and all(consumers.get(name) == 1 for name in (square_out, reduce_out, add_out, sqrt_out))
                and all(name not in graph_outputs for name in (square_out, reduce_out, add_out, sqrt_out))
            )
            if matches:
                output_name = divide.outputs[0]
                scale_name = ""
                consumed = 5
                if index + 5 < len(ops):
                    scale_op = ops[index + 5]
                    if (
                        scale_op.op_type == "Mul"
                        and len(scale_op.inputs) == 2
                        and output_name in scale_op.inputs
                        and consumers.get(output_name) == 1
                        and output_name not in graph_outputs
                    ):
                        candidate = scale_op.inputs[1] if scale_op.inputs[0] == output_name else scale_op.inputs[0]
                        if candidate in initializer_names:
                            scale_name = candidate
                            output_name = scale_op.outputs[0]
                            consumed = 6
                fused.append(
                    _Op(
                        op_type="_RmsNormalization",
                        inputs=[x_name, epsilon_name, scale_name],
                        outputs=[output_name],
                    )
                )
                index += consumed
                continue
        fused.append(ops[index])
        index += 1
    return fused


def _np_to_warp(arr_np: np.ndarray, device: wp.context.Device, requires_grad: bool = False) -> wp.array:
    arr_np = np.ascontiguousarray(arr_np, dtype=np.float32)
    return wp.array(arr_np, dtype=wp.float32, device=device, requires_grad=requires_grad)


class OnnxRuntime:
    """Lightweight ONNX inference engine for graph-capturable MLP policies.

    Args:
        path: Path to an ``.onnx`` file.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.  Defaults to ``1``.
        input_batch_axes: Optional batch-axis override for graph inputs.  If
            an integer is provided, it is applied to every graph input; if a
            dictionary is provided, it maps graph input names to their batch
            axis.  The selected axes are replaced with ``batch_size`` even
            when the ONNX model exported them as fixed dimensions.
        requires_grad: Whether runtime-owned tensors, including initializers
            and intermediate buffers, should allocate gradient storage.  Keep
            this disabled for inference/replay and enable it when computing
            gradients through ONNX runtime outputs.
    """

    def __init__(
        self,
        path: str,
        device: str | wp.Device | None = None,
        batch_size: int = 1,
        input_batch_axes: int | dict[str, int] | None = None,
        requires_grad: bool = False,
    ):
        self._device = parse_device(device)
        self._requires_grad = requires_grad

        onnx, numpy_helper = _require_onnx()
        model = onnx.load(path)
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as exc:
            raise ValueError(f"OnnxRuntime: invalid ONNX model: {exc}") from exc
        graph = model.graph

        self._tensors: dict[str, wp.array] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

        for init in graph.initializer:
            arr_np = numpy_helper.to_array(init).astype(np.float32)
            self._tensors[init.name] = _np_to_warp(arr_np, self._device, requires_grad=self._requires_grad)
            self._shapes[init.name] = tuple(arr_np.shape)

        initializer_names = {init.name for init in graph.initializer}
        self.input_names: list[str] = [inp.name for inp in graph.input if inp.name not in initializer_names]
        self.output_names: list[str] = [out.name for out in graph.output]

        if isinstance(input_batch_axes, dict):
            unknown_inputs = set(input_batch_axes) - set(self.input_names)
            if unknown_inputs:
                raise KeyError(
                    f"OnnxRuntime: input_batch_axes references unknown graph inputs {sorted(unknown_inputs)}"
                )

        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            dims = list(inp.type.tensor_type.shape.dim)
            batch_axis = None
            if input_batch_axes is not None:
                if isinstance(input_batch_axes, dict):
                    batch_axis = input_batch_axes.get(inp.name)
                else:
                    batch_axis = input_batch_axes
                if batch_axis is not None:
                    if batch_axis < 0:
                        batch_axis += len(dims)
                    if batch_axis < 0 or batch_axis >= len(dims):
                        raise ValueError(
                            f"OnnxRuntime: input '{inp.name}' batch axis {batch_axis} is out of range "
                            f"for rank-{len(dims)} input"
                        )
            shape = []
            for axis, d in enumerate(dims):
                if axis == batch_axis:
                    shape.append(batch_size)
                elif d.HasField("dim_value") and d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(batch_size)
            self._shapes[inp.name] = tuple(shape)

        self._ops: list[_Op] = []
        for node in graph.node:
            decoded, all_names = _decode_attrs(node)
            self._ops.append(
                _Op(
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attrs=decoded,
                    attr_names=all_names,
                )
            )
        if not self._requires_grad:
            self._ops = _fuse_inference_ops(self._ops, set(self.output_names), initializer_names)

        self._preallocate_buffers()

    def _preallocate_buffers(self) -> None:
        for op in self._ops:
            handler = _SHAPE_DISPATCH.get(op.op_type)
            if handler is None:
                supported = sorted(name for name in _OP_DISPATCH if not name.startswith("_"))
                raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'.  Supported ops: {supported}")
            handler(op, self._shapes, self._tensors, self._device, self._requires_grad)

    def __call__(self, inputs: dict[str, wp.array]) -> dict[str, wp.array]:
        """Run forward inference.

        Args:
            inputs: Mapping of ONNX input names to Warp arrays already on
                the correct device.  2-D ``wp.array2d`` is the typical case.

        Returns:
            Mapping of ONNX output names to Warp result arrays.
        """
        tensors = self._tensors

        declared_inputs = set(self.input_names)
        for name in inputs:
            if name not in declared_inputs:
                raise KeyError(f"OnnxRuntime: unknown input '{name}'")

        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"OnnxRuntime: missing input '{name}'")
            arr = inputs[name]
            expected_shape = self._shapes[name]
            if tuple(arr.shape) != expected_shape:
                raise ValueError(f"OnnxRuntime: input '{name}' has shape {tuple(arr.shape)}, expected {expected_shape}")
            tensors[name] = arr

        for op in self._ops:
            dispatch = _OP_DISPATCH.get(op.op_type)
            if dispatch is None:
                raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'")
            dispatch(op, tensors, self._shapes, self._device)

        return {name: tensors[name] for name in self.output_names}


def _shape_gemm(op, shapes, tensors, device, requires_grad=False):
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    transA = int(op.attrs.get("transA", 0))
    transB = int(op.attrs.get("transB", 0))
    if transA:
        raise NotImplementedError("OnnxRuntime Gemm: transA=1 is not graph-capturable in this runtime")
    if transB != 1:
        raise NotImplementedError("OnnxRuntime Gemm: only transB=1 policy weights are supported")
    if len(op.inputs) < 3 or not op.inputs[2]:
        raise NotImplementedError("OnnxRuntime Gemm: bias input is required for graph-capturable policy execution")
    if len(A_shape) != 2 or len(B_shape) != 2:
        raise NotImplementedError("OnnxRuntime Gemm: only 2-D tensors are supported")
    M = A_shape[0]
    N = B_shape[0]
    K = A_shape[1]
    if B_shape[1] != K:
        raise ValueError(f"OnnxRuntime Gemm: incompatible shapes {A_shape} and {B_shape}")
    bias_shape = shapes[op.inputs[2]]
    if bias_shape != (N,):
        raise ValueError(f"OnnxRuntime Gemm: bias '{op.inputs[2]}' has shape {bias_shape}, expected {(N,)}")
    out_shape = (M, N)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
    shapes[out_name] = out_shape
    op.attrs["_bias_2d"] = tensors[op.inputs[2]].reshape((N, 1))
    op.attrs["_requires_grad"] = requires_grad


def _shape_elementwise_unary(op, shapes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    if len(in_shape) != 2:
        raise NotImplementedError("OnnxRuntime Elu: only 2-D tensors are supported")
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(in_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
    shapes[out_name] = in_shape


def _shape_elementwise_binary(op, shapes, tensors, device, requires_grad=False):
    lhs_shape = shapes[op.inputs[0]]
    rhs_shape = shapes[op.inputs[1]]
    if len(lhs_shape) not in (1, 2) or len(rhs_shape) not in (1, 2):
        raise NotImplementedError(f"OnnxRuntime {op.op_type}: only 1-D and 2-D tensors are supported")
    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        raise NotImplementedError(f"OnnxRuntime {op.op_type}: at least one input must be 2-D")
    lhs_2d = (1, lhs_shape[0]) if len(lhs_shape) == 1 else lhs_shape
    rhs_2d = (1, rhs_shape[0]) if len(rhs_shape) == 1 else rhs_shape
    for lhs_size, rhs_size in zip(lhs_2d, rhs_2d):
        if lhs_size != rhs_size and lhs_size != 1 and rhs_size != 1:
            raise ValueError(f"OnnxRuntime {op.op_type}: shapes {lhs_shape} and {rhs_shape} do not broadcast")
    out_shape = tuple(max(lhs_size, rhs_size) for lhs_size, rhs_size in zip(lhs_2d, rhs_2d))
    if requires_grad and (lhs_2d != out_shape or rhs_2d != out_shape):
        raise NotImplementedError(f"OnnxRuntime {op.op_type}: broadcast gradients are not supported deterministically")
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
    shapes[out_name] = out_shape
    op.attrs["_lhs_shape_2d"] = lhs_2d
    op.attrs["_rhs_shape_2d"] = rhs_2d
    if op.inputs[0] in tensors and len(lhs_shape) == 1:
        op.attrs["_lhs_view"] = tensors[op.inputs[0]].reshape(lhs_2d)
    if op.inputs[1] in tensors and len(rhs_shape) == 1:
        op.attrs["_rhs_view"] = tensors[op.inputs[1]].reshape(rhs_2d)


def _shape_reduce_mean(op, shapes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    axes = tuple(int(axis) for axis in op.attrs.get("axes", []))
    keepdims = int(op.attrs.get("keepdims", 1))
    if len(in_shape) != 2 or axes not in ((1,), (-1,)) or keepdims != 1:
        raise NotImplementedError("OnnxRuntime ReduceMean: only 2-D row reductions with keepdims=1 are supported")
    out_shape = (in_shape[0], 1)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
    shapes[out_name] = out_shape


def _shape_batch_normalization(op, shapes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime BatchNormalization: deterministic gradients are not supported")
    if len(op.inputs) != 5:
        raise NotImplementedError("OnnxRuntime BatchNormalization: training inputs are not supported")
    in_shape = shapes[op.inputs[0]]
    if len(in_shape) != 2:
        raise NotImplementedError("OnnxRuntime BatchNormalization: only 2-D tensors are supported")
    width = in_shape[1]
    for name in op.inputs[1:]:
        if shapes[name] != (width,):
            raise ValueError(
                f"OnnxRuntime BatchNormalization: parameter '{name}' has shape {shapes[name]}, expected {(width,)}"
            )
    if int(op.attrs.get("training_mode", 0)) != 0:
        raise NotImplementedError("OnnxRuntime BatchNormalization: training mode is not supported")
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(in_shape, dtype=wp.float32, device=device, requires_grad=requires_grad)
    shapes[out_name] = in_shape


def _shape_rms_normalization(op, shapes, tensors, device, requires_grad=False):
    if requires_grad:
        raise RuntimeError("internal inference fusion cannot require gradients")
    shape = shapes[op.inputs[0]]
    if len(shape) != 2:
        raise ValueError(f"OnnxRuntime fused RMS normalization requires a 2-D input, got {shape}")
    if shapes[op.inputs[1]] != (1,):
        raise ValueError("OnnxRuntime fused RMS normalization epsilon must have shape (1,)")
    width = shape[1]
    if op.inputs[2]:
        if shapes[op.inputs[2]] != (width,):
            raise ValueError("OnnxRuntime fused RMS normalization scale has invalid shape")
        op.attrs["_scale"] = tensors[op.inputs[2]]
    else:
        op.attrs["_scale"] = wp.ones(width, dtype=wp.float32, device=device)
    op.attrs["_kernel"] = _create_rms_normalization_kernel(width)
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=wp.float32, device=device)
    shapes[op.outputs[0]] = shape


def _shape_squeeze(op, shapes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    axes = None
    if len(op.inputs) > 1 and op.inputs[1] in tensors:
        axes_tensor = tensors[op.inputs[1]]
        if hasattr(axes_tensor, "numpy"):
            axes = [int(v) for v in axes_tensor.numpy().tolist()]
    if axes is None:
        out_shape = tuple(d for d in in_shape if d != 1)
    else:
        rank = len(in_shape)
        axes_norm = {a if a >= 0 else a + rank for a in axes}
        out_shape = tuple(d for i, d in enumerate(in_shape) if i not in axes_norm)
    if len(out_shape) != 2:
        raise NotImplementedError(
            f"OnnxRuntime Squeeze: only squeezes that produce a 2-D tensor are supported (got {out_shape})"
        )
    shapes[op.outputs[0]] = out_shape
    op.attrs["_out_shape"] = out_shape


def _shape_lstm(op, shapes, tensors, device, requires_grad=False):
    for unsupported in ("activations", "activation_alpha", "activation_beta"):
        if unsupported in op.attr_names:
            raise NotImplementedError(
                f"OnnxRuntime LSTM: attribute '{unsupported}' is not supported "
                f"(only default sigmoid/tanh/tanh activations)"
            )
    if op.attrs.get("clip", 0.0):
        raise NotImplementedError(
            f"OnnxRuntime LSTM: non-default 'clip' attribute is not supported (got {op.attrs['clip']})"
        )
    if op.attrs.get("input_forget", 0):
        raise NotImplementedError(
            f"OnnxRuntime LSTM: non-default 'input_forget' attribute is not supported (got {op.attrs['input_forget']})"
        )

    if len(op.inputs) > 4 and op.inputs[4]:
        raise NotImplementedError("OnnxRuntime LSTM: 'sequence_lens' input is not supported")
    if len(op.inputs) > 7 and op.inputs[7]:
        raise NotImplementedError("OnnxRuntime LSTM: peephole input 'P' is not supported")

    direction = op.attrs.get("direction", "forward")
    if direction not in ("forward", b"forward"):
        raise NotImplementedError("OnnxRuntime LSTM: only forward direction is supported")

    layout = int(op.attrs.get("layout", 0))
    if layout != 0:
        raise NotImplementedError("OnnxRuntime LSTM: layout must be 0 (layout=1 not supported)")

    X_shape = shapes[op.inputs[0]]
    if len(X_shape) != 3:
        raise NotImplementedError("OnnxRuntime LSTM: input X must be 3-D")
    if layout == 0:
        seq_len, batch, input_size = X_shape
    else:
        batch, seq_len, input_size = X_shape
    if seq_len != 1:
        raise NotImplementedError("OnnxRuntime LSTM: only seq_length=1 is supported (single-step inference)")

    W_shape = shapes[op.inputs[1]]
    if len(W_shape) != 3 or W_shape[0] != 1:
        raise NotImplementedError("OnnxRuntime LSTM: only num_directions=1 is supported")
    hidden_size = int(op.attrs.get("hidden_size", W_shape[1] // 4))

    if W_shape != (1, 4 * hidden_size, input_size):
        raise ValueError(f"OnnxRuntime LSTM: W has shape {W_shape}, expected {(1, 4 * hidden_size, input_size)}")

    R_shape = shapes[op.inputs[2]]
    if R_shape != (1, 4 * hidden_size, hidden_size):
        raise ValueError(f"OnnxRuntime LSTM: R has shape {R_shape}, expected {(1, 4 * hidden_size, hidden_size)}")

    W_full = tensors[op.inputs[1]]
    R_full = tensors[op.inputs[2]]
    cache: dict[str, wp.array] = {}
    cache["W"] = W_full.reshape((4 * hidden_size, input_size))
    cache["R"] = R_full.reshape((4 * hidden_size, hidden_size))

    if len(op.inputs) > 3 and op.inputs[3] and op.inputs[3] in tensors:
        B_full = tensors[op.inputs[3]]
        B_shape_in = shapes[op.inputs[3]]
        if B_shape_in != (1, 8 * hidden_size):
            raise ValueError(f"OnnxRuntime LSTM: B has shape {B_shape_in}, expected {(1, 8 * hidden_size)}")
        B_2d = B_full.reshape((8 * hidden_size,))
        cache["Bx"] = B_2d[: 4 * hidden_size]
        cache["Bh"] = B_2d[4 * hidden_size :]
    else:
        cache["Bx"] = wp.zeros(
            4 * hidden_size,
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        cache["Bh"] = wp.zeros(
            4 * hidden_size,
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )

    cache["gates"] = wp.zeros(
        (batch, 4 * hidden_size),
        dtype=wp.float32,
        device=device,
        requires_grad=requires_grad,
    )
    cache["input_size"] = input_size
    cache["hidden_size"] = hidden_size
    cache["batch"] = batch
    cache["layout"] = layout
    op.attrs["_cache"] = cache

    h_buf = wp.zeros(
        (batch, hidden_size),
        dtype=wp.float32,
        device=device,
        requires_grad=requires_grad,
    )
    c_buf = wp.zeros(
        (batch, hidden_size),
        dtype=wp.float32,
        device=device,
        requires_grad=requires_grad,
    )
    cache["h_out"] = h_buf
    cache["c_out"] = c_buf

    if layout == 0:
        Y_shape = (1, 1, batch, hidden_size)
    else:
        Y_shape = (batch, 1, 1, hidden_size)
    Yh_shape = (1, batch, hidden_size)

    if len(op.outputs) > 0 and op.outputs[0]:
        tensors[op.outputs[0]] = h_buf.reshape(Y_shape)
        shapes[op.outputs[0]] = Y_shape
    if len(op.outputs) > 1 and op.outputs[1]:
        tensors[op.outputs[1]] = h_buf.reshape(Yh_shape)
        shapes[op.outputs[1]] = Yh_shape
    if len(op.outputs) > 2 and op.outputs[2]:
        tensors[op.outputs[2]] = c_buf.reshape(Yh_shape)
        shapes[op.outputs[2]] = Yh_shape


def _exec_gemm(op, tensors, shapes, device):
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    bias = tensors[op.inputs[2]]
    out = tensors[op.outputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    beta = float(op.attrs.get("beta", 1.0))
    M = shapes[op.inputs[0]][0]
    N, K = shapes[op.inputs[1]]

    if op.attrs["_requires_grad"]:
        wp.launch(
            _gemm_transb_kernel,
            dim=(M, N),
            inputs=[A, B, bias, out, K, alpha, beta],
            device=device,
        )
    else:
        wp.launch_tiled(
            _GEMM_TRANSB_TILED_KERNEL,
            dim=resolve_dim(config=_GEMM_CONFIG, shape=(M, N), tiled=True),
            inputs=[A, B, op.attrs["_bias_2d"], alpha, beta],
            outputs=[out],
            device=device,
            block_dim=_GEMM_CONFIG.block_dim,
        )


def _exec_elu(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_elu_kernel, dim=shape, inputs=[x, out, alpha], device=device)


def _exec_unary(op, tensors, shapes, device):
    operation = {"Relu": 0, "Tanh": 1, "Sqrt": 2}[op.op_type]
    wp.launch(
        _unary_kernel,
        dim=shapes[op.inputs[0]],
        inputs=[tensors[op.inputs[0]], operation],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_binary(op, tensors, shapes, device):
    lhs = op.attrs.get("_lhs_view", tensors[op.inputs[0]])
    rhs = op.attrs.get("_rhs_view", tensors[op.inputs[1]])
    if len(shapes[op.inputs[0]]) == 1 and "_lhs_view" not in op.attrs:
        lhs = lhs.reshape(op.attrs["_lhs_shape_2d"])
    if len(shapes[op.inputs[1]]) == 1 and "_rhs_view" not in op.attrs:
        rhs = rhs.reshape(op.attrs["_rhs_shape_2d"])
    operation = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3}[op.op_type]
    wp.launch(
        _binary_broadcast_kernel,
        dim=shapes[op.outputs[0]],
        inputs=[lhs, rhs, operation],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_reduce_mean(op, tensors, shapes, device):
    wp.launch(
        _reduce_mean_rows_kernel,
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]]],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_batch_normalization(op, tensors, shapes, device):
    wp.launch(
        _batch_normalization_kernel,
        dim=shapes[op.inputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.inputs[4]],
            float(op.attrs.get("epsilon", 1.0e-5)),
            op.op_type == "_BatchNormalizationRelu",
        ],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_rms_normalization(op, tensors, shapes, device):
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]], tensors[op.inputs[1]], op.attrs["_scale"]],
        outputs=[tensors[op.outputs[0]]],
        device=device,
        block_dim=_GEMM_CONFIG.block_dim,
    )


def _exec_squeeze(op, tensors, shapes, device):
    src = tensors[op.inputs[0]]
    out_shape = op.attrs["_out_shape"]
    tensors[op.outputs[0]] = src.reshape(out_shape)
    shapes[op.outputs[0]] = out_shape


def _exec_lstm(op, tensors, shapes, device):
    cache = op.attrs["_cache"]
    input_size: int = cache["input_size"]
    hidden_size: int = cache["hidden_size"]
    batch: int = cache["batch"]
    layout: int = cache["layout"]

    X = tensors[op.inputs[0]]
    if layout == 0:
        x_t = X.reshape((batch, input_size))
    else:
        x_t = X.reshape((batch, input_size))

    if len(op.inputs) > 5 and op.inputs[5] and op.inputs[5] in tensors:
        h_prev = tensors[op.inputs[5]].reshape((batch, hidden_size))
    else:
        if "h_prev_zero" not in cache:
            cache["h_prev_zero"] = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
        h_prev = cache["h_prev_zero"]
    if len(op.inputs) > 6 and op.inputs[6] and op.inputs[6] in tensors:
        c_prev = tensors[op.inputs[6]].reshape((batch, hidden_size))
    else:
        if "c_prev_zero" not in cache:
            cache["c_prev_zero"] = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
        c_prev = cache["c_prev_zero"]

    gates = cache["gates"]
    h_out = cache["h_out"]
    c_out = cache["c_out"]

    wp.launch(
        _lstm_gates_kernel,
        dim=(batch, 4 * hidden_size),
        inputs=[x_t, h_prev, cache["W"], cache["R"], gates, input_size, hidden_size],
        device=device,
    )
    wp.launch(
        _lstm_cell_update_kernel,
        dim=(batch, hidden_size),
        inputs=[gates, c_prev, cache["Bx"], cache["Bh"], h_out, c_out, hidden_size],
        device=device,
    )


_OP_DISPATCH: dict[str, Any] = {
    "_BatchNormalizationRelu": _exec_batch_normalization,
    "_RmsNormalization": _exec_rms_normalization,
    "Add": _exec_binary,
    "BatchNormalization": _exec_batch_normalization,
    "Div": _exec_binary,
    "Elu": _exec_elu,
    "Gemm": _exec_gemm,
    "LSTM": _exec_lstm,
    "Mul": _exec_binary,
    "ReduceMean": _exec_reduce_mean,
    "Relu": _exec_unary,
    "Sqrt": _exec_unary,
    "Squeeze": _exec_squeeze,
    "Sub": _exec_binary,
    "Tanh": _exec_unary,
}

_SHAPE_DISPATCH: dict[str, Any] = {
    "_BatchNormalizationRelu": _shape_batch_normalization,
    "_RmsNormalization": _shape_rms_normalization,
    "Add": _shape_elementwise_binary,
    "BatchNormalization": _shape_batch_normalization,
    "Div": _shape_elementwise_binary,
    "Elu": _shape_elementwise_unary,
    "Gemm": _shape_gemm,
    "LSTM": _shape_lstm,
    "Mul": _shape_elementwise_binary,
    "ReduceMean": _shape_reduce_mean,
    "Relu": _shape_elementwise_unary,
    "Sqrt": _shape_elementwise_unary,
    "Squeeze": _shape_squeeze,
    "Sub": _shape_elementwise_binary,
    "Tanh": _shape_elementwise_unary,
}
