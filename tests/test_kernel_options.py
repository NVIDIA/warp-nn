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

"""Check the options of the library's kernel decorators."""

import pytest

import ast
import pathlib


_PACKAGE = pathlib.Path(__file__).parents[1] / "warp_nn"


def _is_kernel_decorator(node: ast.expr) -> bool:
    """Whether a decorator is ``@wp.kernel`` or ``@wp.kernel(...)``."""
    target = node.func if isinstance(node, ast.Call) else node
    return ast.unparse(target) == "wp.kernel"


def _kernel_decorators() -> list:
    """Collect the kernel decorators of the library as ``(location, decorator)`` test parameters."""
    params = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.FunctionDef):
                for decorator in filter(_is_kernel_decorator, node.decorator_list):
                    location = f"{path.relative_to(_PACKAGE.parent)}:{decorator.lineno}"
                    params.append(pytest.param(decorator, id=location))
    return params


def test_kernels_found():
    assert len(_kernel_decorators()) > 0


@pytest.mark.parametrize("decorator", _kernel_decorators())
def test_grid_stride_disabled(decorator):
    # kernels are launched with one thread per launch index, so the grid-stride loop never iterates
    keywords = {
        keyword.arg: keyword.value for keyword in (decorator.keywords if isinstance(decorator, ast.Call) else [])
    }
    assert "grid_stride" in keywords, f"Missing 'grid_stride=False' in '@{ast.unparse(decorator)}'"
    assert ast.literal_eval(keywords["grid_stride"]) is False
