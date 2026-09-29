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

"""Run the docstring examples of the library."""

import pytest

import doctest
import inspect
from functools import reduce
from types import ModuleType

import numpy as np
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime import onnx_runtime, onnx_runtime_v2


wp.init()


def _owner(module: ModuleType, test: doctest.DocTest) -> str:
    """Name of the class or module that owns a docstring example (classes and modules own their docstring)."""
    path = test.name.removeprefix(module.__name__).lstrip(".")
    documented = reduce(getattr, path.split("."), module) if path else module
    return test.name if inspect.isclass(documented) or inspect.ismodule(documented) else test.name.rpartition(".")[0]


def _doctests(*modules: ModuleType) -> list:
    """Collect the owners (classes or modules) of the docstring examples of the given modules as ``(module, owner)``
    test parameters."""
    return [
        pytest.param(module, owner, id=owner.removeprefix("warp_nn."))
        for module in modules
        for owner in dict.fromkeys(
            _owner(module, test) for test in doctest.DocTestFinder().find(module) if test.examples
        )
    ]


def _skip_if_unavailable(device: str) -> None:
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")


def _run_doctests(
    module: ModuleType,
    owner: str,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the docstring examples of a class or module on the given device.

    The examples share the same globals, so that later examples can use the names (e.g. imports and instances)
    defined by earlier ones. They are run alphabetically, but with the class (or module) and ``__init__`` examples
    first, and stop at the first failing example (since the following ones may depend on it).
    """
    # silence the Warp messages printed while loading modules, which would be part of the examples output
    monkeypatch.setattr(wp.config, "log_level", wp.LOG_WARNING)
    tests = sorted(
        (test for test in doctest.DocTestFinder().find(module) if test.examples and _owner(module, test) == owner),
        key=lambda test: (test.name != owner, not test.name.endswith(".__init__"), test.name),
    )
    globs = tests[0].globs
    runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    report = []
    try:
        with wp.ScopedDevice(device):
            for test in tests:
                test.globs = globs
                results = runner.run(test, out=report.append, clear_globs=False)
                # report the progress (shown when the output is not captured, e.g. with `pytest -s`)
                print(
                    f"\n  |-- {test.name}: {results.attempted - results.failed}/{results.attempted} passed",
                    end="",
                )
                if results.failed:
                    break
    finally:
        globs.clear()
        # end the progress report, so that the pytest outcome is written on its own line
        print()
    assert runner.failures == 0, "".join(report)


def _generate_onnx_file(path) -> None:
    """Save an ONNX MLP policy.

    The policy maps ``obs`` of shape ``(batch, 6)`` to ``actions`` of shape ``(batch, 3)``.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    constants = {
        "w1": rng.uniform(-0.5, 0.5, (8, 6)).astype(np.float32),
        "b1": rng.uniform(-0.5, 0.5, (8,)).astype(np.float32),
        "w2": rng.uniform(-0.5, 0.5, (3, 8)).astype(np.float32),
        "b2": rng.uniform(-0.5, 0.5, (3,)).astype(np.float32),
    }
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["obs", "w1", "b1"], ["h1"], transB=1),
            helper.make_node("Elu", ["h1"], ["a1"]),
            helper.make_node("Gemm", ["a1", "w2", "b2"], ["h2"], transB=1),
            helper.make_node("Tanh", ["h2"], ["actions"]),
        ],
        "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ("batch", 6))],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ("batch", 3))],
        initializer=[numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module, owner", _doctests(onnx_runtime, onnx_runtime_v2))
def test_runtime(tmp_path, monkeypatch, device, module, owner):
    _skip_if_unavailable(device)
    _generate_onnx_file(tmp_path / "policy.onnx")
    monkeypatch.chdir(tmp_path)
    _run_doctests(module, owner, device, monkeypatch)
