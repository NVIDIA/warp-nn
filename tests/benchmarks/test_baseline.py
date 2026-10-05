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

import importlib
import pathlib
import shutil
import subprocess
import sys

import warp as wp

from benchmarks import _baseline


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")


def _git(repository: pathlib.Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repository,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repository(tmp_path):
    # repository with a package whose committed version differs from its working tree version
    repository = tmp_path / "repository"
    package = repository / "fake_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "values.py").write_text("VALUE = 'committed'\n")
    (package / "nn.py").write_text("from fake_pkg.values import VALUE\nimport fake_pkg\n")
    (package / "kernels.py").write_text(
        "import warp as wp\n\n\n@wp.kernel\ndef kernel(a: wp.array(dtype=float)):\n    a[wp.tid()] = 1.0\n"
    )
    (repository / "outside.py").write_text("")
    _git(repository, "init", "--quiet")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "initial")
    _git(repository, "branch", "baseline-branch")
    (package / "values.py").write_text("VALUE = 'working tree'\n")
    return repository


def test_rewrite_imports(tmp_path):
    source = (
        "import warp_nn\n"
        "import warp_nn.nn as nn\n"
        "from warp_nn.modules import Module\n"
        "x = warp_nn.utils\n"
        "warp_nn_other = 1  # WARP_NN and my_warp_nn are other names\n"
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "module.py").write_text(source)
    (tmp_path / "data.txt").write_text("import warp_nn\n")
    _baseline.rewrite_imports(tmp_path, old="warp_nn", new="warp_nn_baseline")
    assert (tmp_path / "sub" / "module.py").read_text() == (
        "import warp_nn_baseline\n"
        "import warp_nn_baseline.nn as nn\n"
        "from warp_nn_baseline.modules import Module\n"
        "x = warp_nn_baseline.utils\n"
        "warp_nn_other = 1  # WARP_NN and my_warp_nn are other names\n"
    )
    # only Python source files are rewritten
    assert (tmp_path / "data.txt").read_text() == "import warp_nn\n"


def test_resolve_commit(repository):
    commit = _baseline.resolve_commit("baseline-branch", repository=repository)
    assert len(commit) == 40
    with pytest.raises(RuntimeError, match="git rev-parse"):
        _baseline.resolve_commit("missing-branch", repository=repository)


def test_export_package(repository, tmp_path, monkeypatch):
    destination = tmp_path / "export"
    destination.mkdir()
    directory = _baseline.export_package(
        "baseline-branch", repository=repository, destination=destination, package="fake_pkg", alias="fake_pkg_alias"
    )
    assert directory == destination / "fake_pkg_alias"
    # only the package is exported, with its committed content and its imports rewritten
    assert sorted(path.name for path in destination.iterdir()) == ["fake_pkg_alias"]
    assert (directory / "nn.py").read_text() == "from fake_pkg_alias.values import VALUE\nimport fake_pkg_alias\n"
    monkeypatch.syspath_prepend(str(destination))
    try:
        assert importlib.import_module("fake_pkg_alias.nn").VALUE == "committed"
    finally:
        for name in [name for name in sys.modules if name.startswith("fake_pkg_alias")]:
            del sys.modules[name]


def test_import_baseline_already_imported(repository, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, _baseline.ALIAS, object())
    with pytest.raises(RuntimeError, match="already imported"):
        _baseline.import_baseline("baseline-branch", repository=repository, destination=tmp_path)


def test_import_baseline_missing_reference(repository, tmp_path):
    with pytest.raises(RuntimeError, match="git rev-parse"):
        _baseline.import_baseline("missing-branch", repository=repository, destination=tmp_path)


def test_import_baseline(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # restore the module search path afterwards
    try:
        nn, commit = _baseline.import_baseline(
            "baseline-branch", repository=repository, destination=tmp_path, package="fake_pkg", alias="fake_pkg_alias"
        )
        assert nn.__name__ == "fake_pkg_alias.nn"
        importlib.import_module("fake_pkg_alias.kernels")  # registers a Warp module named after the Python module
        assert "fake_pkg_alias.kernels" in wp._src.context.user_modules
        assert nn.VALUE == "committed"
        assert commit == _baseline.resolve_commit("baseline-branch", repository=repository)
        assert str(tmp_path) in sys.path
    finally:
        _baseline.unload_baseline(destination=tmp_path, alias="fake_pkg_alias")
    assert str(tmp_path) not in sys.path
    assert not [name for name in sys.modules if name.startswith("fake_pkg_alias")]
    assert "fake_pkg_alias.kernels" not in wp._src.context.user_modules
    # it can be imported again
    try:
        nn, _ = _baseline.import_baseline(
            "baseline-branch",
            repository=repository,
            destination=tmp_path / "again",
            package="fake_pkg",
            alias="fake_pkg_alias",
        )
        assert nn.VALUE == "committed"
    finally:
        _baseline.unload_baseline(destination=tmp_path / "again", alias="fake_pkg_alias")


def test_import_baseline_failure_cleanup(repository, tmp_path):
    # the exported package cannot be imported: the import is undone
    (repository / "fake_pkg" / "nn.py").write_text("raise ImportError('broken')\n")
    _git(repository, "commit", "--quiet", "-am", "broken")
    with pytest.raises(ImportError, match="broken"):
        _baseline.import_baseline(
            "HEAD", repository=repository, destination=tmp_path, package="fake_pkg", alias="fake_pkg_alias"
        )
    assert str(tmp_path) not in sys.path
    assert not [name for name in sys.modules if name.startswith("fake_pkg_alias")]
