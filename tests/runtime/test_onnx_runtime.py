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

import tempfile
from pathlib import Path


pytest.importorskip("onnx")

import onnx
from onnx import TensorProto, helper, numpy_helper

import numpy as np
import warp as wp

from tests.utilities import check_arrays, is_device_available
from warp_nn.runtime import OnnxRuntime


def _node_attrs(node) -> dict[str, float | int]:
    attrs: dict[str, float | int] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
    return attrs


def _run_numpy_mlp(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {name: arr.astype(np.float32) for name, arr in feeds.items()}
    values.update({init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer})

    for node in model.graph.node:
        attrs = _node_attrs(node)
        if node.op_type == "Gemm":
            A = values[node.input[0]]
            B = values[node.input[1]]
            bias = values[node.input[2]]
            if int(attrs.get("transA", 0)) != 0 or int(attrs.get("transB", 0)) != 1:
                raise NotImplementedError("policy reference only supports transA=0, transB=1")
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            values[node.output[0]] = (alpha * (A @ B.T) + beta * bias).astype(np.float32)
        elif node.op_type == "Elu":
            x = values[node.input[0]]
            alpha = float(attrs.get("alpha", 1.0))
            values[node.output[0]] = np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)
        else:
            raise NotImplementedError(f"unsupported op in policy reference: {node.op_type}")

    return {out.name: values[out.name] for out in model.graph.output}


def _finite_difference_gradient(fwd, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float32)
    x_flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)
    for i in range(x_flat.size):
        xp = x_flat.copy()
        xm = x_flat.copy()
        xp[i] += eps
        xm[i] -= eps
        grad_flat[i] = (fwd(xp.reshape(x.shape)) - fwd(xm.reshape(x.shape))) / (2.0 * eps)
    return grad


def _runtime_input_gradient(
    rt: OnnxRuntime,
    feeds: dict[str, np.ndarray],
    *,
    input_name: str,
    output_name: str,
    seed: np.ndarray,
    device: str,
) -> np.ndarray:
    inputs = {
        name: wp.array(arr, dtype=wp.float32, device=device, requires_grad=name == input_name)
        for name, arr in feeds.items()
    }
    seed_wp = wp.array(seed, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        out = rt(inputs)[output_name]
    tape.backward(grads={out: seed_wp})

    return inputs[input_name].grad.numpy().copy()


def _build_mlp_policy_model(
    layer_sizes: tuple[int, ...],
    *,
    batch: int = 1,
    seed: int = 0,
) -> onnx.ModelProto:
    """Build a multi-layer Gemm/Elu policy network."""
    rng = np.random.default_rng(seed)
    nodes = []
    initializers = []
    in_dim = layer_sizes[0]
    x_vi = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [batch, in_dim])
    prev = "observation"

    for i, out_dim in enumerate(layer_sizes[1:]):
        W = (rng.standard_normal((out_dim, in_dim)) * 0.3).astype(np.float32)
        b = (rng.standard_normal((out_dim,)) * 0.05).astype(np.float32)
        w_name, b_name = f"W{i}", f"b{i}"
        initializers.extend(
            [
                numpy_helper.from_array(W, name=w_name),
                numpy_helper.from_array(b, name=b_name),
            ]
        )
        is_last = i == len(layer_sizes) - 2
        hidden = "action" if is_last else f"hidden{i}"
        nodes.append(helper.make_node("Gemm", [prev, w_name, b_name], [hidden], alpha=1.0, beta=1.0, transB=1))
        if not is_last:
            activated = f"activated{i}"
            nodes.append(helper.make_node("Elu", [hidden], [activated], alpha=1.0))
            prev = activated
        else:
            prev = hidden
        in_dim = out_dim

    y_vi = helper.make_tensor_value_info("action", TensorProto.FLOAT, [batch, layer_sizes[-1]])
    graph = helper.make_graph(nodes, "mlp_policy", [x_vi], [y_vi], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _build_general_ops_model(batch: int, input_size: int, output_size: int, seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    arrays = {
        "weight": (rng.standard_normal((output_size, input_size)) * 0.3).astype(np.float32),
        "bias": (rng.standard_normal(output_size) * 0.05).astype(np.float32),
        "scale": rng.uniform(0.5, 1.5, output_size).astype(np.float32),
        "bn_bias": (rng.standard_normal(output_size) * 0.05).astype(np.float32),
        "mean": (rng.standard_normal(output_size) * 0.1).astype(np.float32),
        "variance": rng.uniform(0.5, 1.5, output_size).astype(np.float32),
        "epsilon": np.asarray([1.0e-6], dtype=np.float32),
        "rms_scale": rng.uniform(0.8, 1.2, output_size).astype(np.float32),
        "zero": np.asarray([0.0], dtype=np.float32),
    }
    nodes = [
        helper.make_node("Gemm", ["input", "weight", "bias"], ["linear"], transB=1),
        helper.make_node(
            "BatchNormalization",
            ["linear", "scale", "bn_bias", "mean", "variance"],
            ["normalized"],
            epsilon=1.0e-5,
        ),
        helper.make_node("Relu", ["normalized"], ["activated"]),
        helper.make_node("Mul", ["activated", "activated"], ["squared"]),
        helper.make_node("ReduceMean", ["squared"], ["mean_square"], axes=[1], keepdims=1),
        helper.make_node("Add", ["mean_square", "epsilon"], ["mean_square_epsilon"]),
        helper.make_node("Sqrt", ["mean_square_epsilon"], ["root"]),
        helper.make_node("Div", ["activated", "root"], ["unit"]),
        helper.make_node("Mul", ["unit", "rms_scale"], ["scaled_unit"]),
        helper.make_node("Sub", ["scaled_unit", "zero"], ["shifted"]),
        helper.make_node("Tanh", ["shifted"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "general_ops",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, input_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, output_size])],
        [numpy_helper.from_array(value, name=name) for name, value in arrays.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _lstm_step_reference(
    x: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    Bx: np.ndarray,
    Bh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    H = h_prev.shape[1]
    gates = x @ W.T + h_prev @ R.T + Bx + Bh
    g_i = 1.0 / (1.0 + np.exp(-gates[:, 0 * H : 1 * H]))
    g_o = 1.0 / (1.0 + np.exp(-gates[:, 1 * H : 2 * H]))
    g_f = 1.0 / (1.0 + np.exp(-gates[:, 2 * H : 3 * H]))
    g_c = np.tanh(gates[:, 3 * H : 4 * H])
    c_new = g_f * c_prev + g_i * g_c
    h_new = g_o * np.tanh(c_new)
    return h_new.astype(np.float32), c_new.astype(np.float32)


def _build_lstm_step_model(batch: int, input_size: int, hidden_size: int, seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    H = hidden_size
    W = (rng.standard_normal((1, 4 * H, input_size)) * 0.3).astype(np.float32)
    R = (rng.standard_normal((1, 4 * H, H)) * 0.3).astype(np.float32)
    B = (rng.standard_normal((1, 8 * H)) * 0.05).astype(np.float32)
    Wd = (rng.standard_normal((1, H)) * 0.3).astype(np.float32)
    bd = np.zeros((1,), dtype=np.float32)

    x_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, batch, input_size])
    h_in = helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, batch, H])
    c_in = helper.make_tensor_value_info("c_in", TensorProto.FLOAT, [1, batch, H])
    y_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, 1])
    h_out_v = helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, batch, H])
    c_out_v = helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1, batch, H])

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(R, name="R"),
        numpy_helper.from_array(B, name="B"),
        numpy_helper.from_array(Wd, name="Wd"),
        numpy_helper.from_array(bd, name="bd"),
        numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="squeeze_axes"),
    ]
    nodes = [
        helper.make_node(
            "LSTM",
            ["input", "W", "R", "B", "", "h_in", "c_in"],
            ["Y", "h_out", "c_out"],
            hidden_size=H,
        ),
        helper.make_node("Squeeze", ["Y", "squeeze_axes"], ["Y_2d"]),
        helper.make_node("Gemm", ["Y_2d", "Wd", "bd"], ["output"], alpha=1.0, beta=1.0, transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "lstm_step",
        [x_in, h_in, c_in],
        [y_out, h_out_v, c_out_v],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


@pytest.mark.parametrize(
    "node",
    [
        helper.make_node("Gemm", [], ["out"], transB=1),
        helper.make_node("Gemm", ["missing", "B", "bias"], ["out"], transB=1),
    ],
    ids=["missing-inputs", "undefined-input"],
)
def test_rejects_invalid_model(node):
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    model = helper.make_model(helper.make_graph([node], "invalid", [], [output]))

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        with pytest.raises(ValueError, match="OnnxRuntime: invalid ONNX model"):
            OnnxRuntime(str(path), device="cpu")
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_mlp_policy(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _build_mlp_policy_model((48, 128, 64, 12), batch=1, seed=20260430)
    rng = np.random.default_rng(20260430)
    obs = rng.standard_normal((1, 48)).astype(np.float32)
    expected = _run_numpy_mlp(model, {"observation": obs})

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=1)
        out = rt({"observation": wp.array(obs, dtype=wp.float32, device=device)})
        for name, expected_arr in expected.items():
            check_arrays(
                out[name],
                wp.array(expected_arr, dtype=wp.float32, device=device),
                rtol=1e-3,
                atol=1e-4,
            )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_mlp_policy_input_gradients(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _build_mlp_policy_model((3, 4, 1), batch=1, seed=20260716)
    rng = np.random.default_rng(20260716)
    obs = rng.standard_normal((1, 3)).astype(np.float32)
    seed = np.ones((1, 1), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=1, requires_grad=True)
        actual = _runtime_input_gradient(
            rt,
            {"observation": obs},
            input_name="observation",
            output_name="action",
            seed=seed,
            device=device,
        )

        def fwd(x):
            out = _run_numpy_mlp(model, {"observation": x})["action"]
            return float(np.sum(out * seed))

        expected = _finite_difference_gradient(fwd, obs)
        check_arrays(
            wp.array(actual, dtype=wp.float32, device=device),
            wp.array(expected, dtype=wp.float32, device=device),
            rtol=1e-2,
            atol=1e-3,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_graph_capture_replay(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _build_mlp_policy_model((16, 8, 4), batch=1, seed=123)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=1)
        rng = np.random.default_rng(123)
        obs_np = rng.standard_normal((1, 16)).astype(np.float32)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)

        rt({"observation": obs})

        wp.capture_begin(device=device)
        try:
            out = rt({"observation": obs})
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise

        replay_obs_np = rng.standard_normal((1, 16)).astype(np.float32)
        obs.assign(replay_obs_np)
        wp.capture_launch(graph)

        expected = _run_numpy_mlp(model, {"observation": replay_obs_np})
        for name, expected_arr in expected.items():
            check_arrays(
                out[name],
                wp.array(expected_arr, dtype=wp.float32, device=device),
                rtol=1e-3,
                atol=1e-4,
            )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_general_ops_graph_capture_is_deterministic(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, output_size = 65, 16, 9
    model = _build_general_ops_model(batch, input_size, output_size, seed=31)
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    rng = np.random.default_rng(31)
    input_np = rng.standard_normal((batch, input_size)).astype(np.float32)

    def reference(x, output="output"):
        linear = x @ values["weight"].T + values["bias"]
        normalized = (linear - values["mean"]) / np.sqrt(values["variance"] + 1.0e-5)
        activated = np.maximum(normalized * values["scale"] + values["bn_bias"], 0.0)
        root = np.sqrt(np.mean(activated * activated, axis=1, keepdims=True) + values["epsilon"])
        unit = activated / root
        if output == "unit":
            return unit.astype(np.float32)
        return np.tanh(unit * values["rms_scale"] - values["zero"]).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="deterministic gradients"):
            OnnxRuntime(str(path), device=device, batch_size=batch, requires_grad=True)
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        op_types = {op.op_type for op in rt._ops}
        assert {"_BatchNormalizationRelu", "_RmsNormalization"} <= op_types
        assert "Mul" not in op_types
        owned_ptrs = {name: int(value.ptr) for name, value in rt._tensors.items()}
        input_wp = wp.array(input_np, dtype=wp.float32, device=device)
        outputs = rt({"input": input_wp})
        check_arrays(
            outputs["output"],
            wp.array(reference(input_np), dtype=wp.float32, device=device),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        wp.capture_begin(device=device)
        try:
            outputs = rt({"input": input_wp})
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise

        replay_input = rng.standard_normal((batch, input_size)).astype(np.float32)
        input_wp.assign(replay_input)
        wp.capture_launch(graph)
        first = outputs["output"].numpy().copy()
        wp.capture_launch(graph)
        second = outputs["output"].numpy().copy()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, reference(replay_input), rtol=1.0e-5, atol=1.0e-6)
        assert owned_ptrs == {name: int(rt._tensors[name].ptr) for name in owned_ptrs}

        model.graph.output.extend(
            [
                helper.make_tensor_value_info("normalized", TensorProto.FLOAT, [batch, output_size]),
                helper.make_tensor_value_info("squared", TensorProto.FLOAT, [batch, output_size]),
            ]
        )
        onnx.save(model, str(path))
        unfused_rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        assert all(not op.op_type.startswith("_") for op in unfused_rt._ops)
        unfused_output = unfused_rt({"input": input_wp})["output"]
        np.testing.assert_allclose(unfused_output.numpy(), reference(replay_input), rtol=1.0e-5, atol=1.0e-6)

        observable_unit_model = _build_general_ops_model(batch, input_size, output_size, seed=31)
        observable_unit_model.graph.output.append(
            helper.make_tensor_value_info("unit", TensorProto.FLOAT, [batch, output_size])
        )
        onnx.checker.check_model(observable_unit_model)
        onnx.save(observable_unit_model, str(path))
        observable_unit_rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        observable_op_types = {op.op_type for op in observable_unit_rt._ops}
        assert "_RmsNormalization" in observable_op_types
        assert "Mul" in observable_op_types
        observable_outputs = observable_unit_rt({"input": input_wp})
        np.testing.assert_allclose(
            observable_outputs["unit"].numpy(), reference(replay_input, "unit"), rtol=1.0e-5, atol=1.0e-6
        )
        np.testing.assert_allclose(
            observable_outputs["output"].numpy(), reference(replay_input), rtol=1.0e-5, atol=1.0e-6
        )

        dynamic_scale_model = _build_general_ops_model(batch, input_size, output_size, seed=31)
        scale_index = next(
            i for i, item in enumerate(dynamic_scale_model.graph.initializer) if item.name == "rms_scale"
        )
        del dynamic_scale_model.graph.initializer[scale_index]
        dynamic_scale_model.graph.input.append(
            helper.make_tensor_value_info("rms_scale", TensorProto.FLOAT, [output_size])
        )
        onnx.checker.check_model(dynamic_scale_model)
        onnx.save(dynamic_scale_model, str(path))
        dynamic_scale_rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        dynamic_op_types = {op.op_type for op in dynamic_scale_rt._ops}
        assert "_RmsNormalization" in dynamic_op_types
        assert "Mul" in dynamic_op_types
        dynamic_scale_output = dynamic_scale_rt(
            {
                "input": input_wp,
                "rms_scale": wp.array(values["rms_scale"], dtype=wp.float32, device=device),
            }
        )["output"]
        np.testing.assert_allclose(dynamic_scale_output.numpy(), reference(replay_input), rtol=1.0e-5, atol=1.0e-6)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_rejects_unsupported_ops(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = helper.make_model(
        helper.make_graph(
            nodes=[helper.make_node("Sigmoid", ["A"], ["Y"])],
            name="reject",
            inputs=[helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="unsupported op 'Sigmoid'"):
            OnnxRuntime(str(path), device=device)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_rejects_unsupported_op_variants(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model = _build_general_ops_model(2, 4, 4)
        del model.graph.node[0].attribute[:]
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="only transB=1"):
            OnnxRuntime(str(path), device=device, batch_size=2)

        model = _build_general_ops_model(2, 4, 4)
        epsilon = next(item for item in model.graph.initializer if item.name == "epsilon")
        epsilon.CopyFrom(numpy_helper.from_array(np.asarray([[1.0e-6]], dtype=np.float32), name="epsilon"))
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        with pytest.raises(ValueError, match=r"epsilon must have shape \(1,\)"):
            OnnxRuntime(str(path), device=device, batch_size=2)

        model = helper.make_model(
            helper.make_graph(
                nodes=[helper.make_node("Add", ["A", "B"], ["Y"])],
                name="reject_1d_binary",
                inputs=[
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [4]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [4]),
                ],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="at least one input must be 2-D"):
            OnnxRuntime(str(path), device=device)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_single_step(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 4, 2, 8
    model = _build_lstm_step_model(batch, input_size, hidden_size, seed=20260430)

    W = numpy_helper.to_array(model.graph.initializer[0])[0]
    R = numpy_helper.to_array(model.graph.initializer[1])[0]
    B_full = numpy_helper.to_array(model.graph.initializer[2])[0]
    Wd = numpy_helper.to_array(model.graph.initializer[3])[0]
    bd = numpy_helper.to_array(model.graph.initializer[4])
    Bx, Bh = B_full[: 4 * hidden_size], B_full[4 * hidden_size :]

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((1, batch, input_size)).astype(np.float32)
    h_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1
    c_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1

    h_ref, c_ref = _lstm_step_reference(x_np[0], h_np[0], c_np[0], W, R, Bx, Bh)
    expected_out = (h_ref @ Wd.T + bd).reshape(batch, 1).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        out = rt(
            {
                "input": wp.array(x_np, dtype=wp.float32, device=device),
                "h_in": wp.array(h_np, dtype=wp.float32, device=device),
                "c_in": wp.array(c_np, dtype=wp.float32, device=device),
            }
        )
        check_arrays(
            out["output"],
            wp.array(expected_out, dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
        check_arrays(
            out["h_out"],
            wp.array(h_ref.reshape(1, batch, hidden_size), dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
        check_arrays(
            out["c_out"],
            wp.array(c_ref.reshape(1, batch, hidden_size), dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_input_gradients(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 1, 3, 4
    model = _build_lstm_step_model(batch, input_size, hidden_size, seed=20260716)

    W = numpy_helper.to_array(model.graph.initializer[0])[0]
    R = numpy_helper.to_array(model.graph.initializer[1])[0]
    B_full = numpy_helper.to_array(model.graph.initializer[2])[0]
    Wd = numpy_helper.to_array(model.graph.initializer[3])[0]
    bd = numpy_helper.to_array(model.graph.initializer[4])
    Bx, Bh = B_full[: 4 * hidden_size], B_full[4 * hidden_size :]

    rng = np.random.default_rng(16)
    x_np = rng.standard_normal((1, batch, input_size)).astype(np.float32)
    h_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1
    c_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1
    seed = np.ones((batch, 1), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch, requires_grad=True)
        actual = _runtime_input_gradient(
            rt,
            {"input": x_np, "h_in": h_np, "c_in": c_np},
            input_name="input",
            output_name="output",
            seed=seed,
            device=device,
        )

        def fwd(x):
            h_ref, _ = _lstm_step_reference(x[0], h_np[0], c_np[0], W, R, Bx, Bh)
            out = (h_ref @ Wd.T + bd).reshape(batch, 1).astype(np.float32)
            return float(np.sum(out * seed))

        expected = _finite_difference_gradient(fwd, x_np)
        check_arrays(
            wp.array(actual, dtype=wp.float32, device=device),
            wp.array(expected, dtype=wp.float32, device=device),
            rtol=1e-2,
            atol=1e-3,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_graph_capture_replay(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 2, 2, 4
    model = _build_lstm_step_model(batch, input_size, hidden_size, seed=4242)

    W = numpy_helper.to_array(model.graph.initializer[0])[0]
    R = numpy_helper.to_array(model.graph.initializer[1])[0]
    B_full = numpy_helper.to_array(model.graph.initializer[2])[0]
    Wd = numpy_helper.to_array(model.graph.initializer[3])[0]
    bd = numpy_helper.to_array(model.graph.initializer[4])
    Bx, Bh = B_full[: 4 * hidden_size], B_full[4 * hidden_size :]

    rng = np.random.default_rng(9)
    x_np = rng.standard_normal((1, batch, input_size)).astype(np.float32)
    h_np = np.zeros((1, batch, hidden_size), dtype=np.float32)
    c_np = np.zeros((1, batch, hidden_size), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        x_wp = wp.array(x_np, dtype=wp.float32, device=device)
        h_wp = wp.array(h_np, dtype=wp.float32, device=device)
        c_wp = wp.array(c_np, dtype=wp.float32, device=device)

        rt({"input": x_wp, "h_in": h_wp, "c_in": c_wp})

        wp.capture_begin(device=device)
        try:
            out = rt({"input": x_wp, "h_in": h_wp, "c_in": c_wp})
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise

        replay_x = rng.standard_normal((1, batch, input_size)).astype(np.float32)
        x_wp.assign(replay_x)
        wp.capture_launch(graph)

        h_ref, _ = _lstm_step_reference(replay_x[0], h_np[0], c_np[0], W, R, Bx, Bh)
        expected = (h_ref @ Wd.T + bd).reshape(batch, 1).astype(np.float32)
        check_arrays(
            out["output"],
            wp.array(expected, dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
    finally:
        path.unlink(missing_ok=True)
