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

"""Baseline warp-nn implementation, exported from a git reference and imported side by side with the current one.

The package is exported (``git archive``) under another name, and its absolute imports are rewritten accordingly,
so that both implementations can be imported (and benchmarked interleaved) in the same process.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pathlib
import re
import subprocess
import sys
import tarfile
from types import ModuleType

import warp as wp


PACKAGE = "warp_nn"
ALIAS = "warp_nn_baseline"


def _git(*args: str, repository: pathlib.Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as e:
        message = e.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Command 'git {' '.join(args)}' failed: {message}") from e


def resolve_commit(ref: str, *, repository: pathlib.Path) -> str:
    """Resolve a git reference (e.g. a branch name) to a commit hash.

    :param ref: The git reference.
    :param repository: The git repository directory.

    :return: The commit hash.

    :raises RuntimeError: If the reference cannot be resolved to a commit.
    """
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}", repository=repository).decode().strip()


def rewrite_imports(directory: pathlib.Path, *, old: str, new: str) -> None:
    """Rename a package in all the Python source files of a directory.

    All the occurrences of the name, as a whole word, are rewritten: not only in the (absolute) imports, but also in
    e.g. strings (such as ``importlib.metadata.version("warp_nn")``, which then falls back to an unknown version).

    :param directory: The directory containing the Python source files (searched recursively).
    :param old: The old package name.
    :param new: The new package name.
    """
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    for path in directory.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        rewritten = pattern.sub(new, source)
        if rewritten != source:
            path.write_text(rewritten, encoding="utf-8")


def export_package(
    commit: str, *, repository: pathlib.Path, destination: pathlib.Path, package: str = PACKAGE, alias: str = ALIAS
) -> pathlib.Path:
    """Export a package from a git commit under another name.

    :param commit: The git commit (or any other git reference).
    :param repository: The git repository directory.
    :param destination: The directory where the package is exported.
    :param package: The package name (directory in the repository root).
    :param alias: The name of the exported package.

    :return: The directory of the exported package.
    """
    archive = _git("archive", "--format=tar", commit, package, repository=repository)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # the "data" filter rejects unsafe members (e.g. absolute paths), when available (Python >= 3.10.12)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(destination, **kwargs)
    directory = destination / alias
    (destination / package).rename(directory)
    rewrite_imports(directory, old=package, new=alias)
    return directory


def import_baseline(
    ref: str, *, repository: pathlib.Path, destination: pathlib.Path, package: str = PACKAGE, alias: str = ALIAS
) -> tuple[ModuleType, str]:
    """Export the warp-nn package from a git reference under another name, and import its ``nn`` namespace.

    .. important::

        The destination directory must exist while the baseline is used,
        since the kernels' source code is read when the modules are instantiated.

    :param ref: The git reference (e.g. a branch name).
    :param repository: The git repository directory.
    :param destination: The directory where the package is exported (and added to the module search path).
    :param package: The package name (directory in the repository root).
    :param alias: The name of the exported package.

    :return: The ``nn`` namespace of the baseline implementation, and the resolved commit hash.

    :raises RuntimeError: If the reference cannot be resolved, or the baseline package is already imported.
    """
    if alias in sys.modules:
        raise RuntimeError(f"Package '{alias}' is already imported")
    commit = resolve_commit(ref, repository=repository)
    export_package(commit, repository=repository, destination=destination, package=package, alias=alias)
    sys.path.insert(0, str(destination))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(f"{alias}.nn"), commit
    except BaseException:
        unload_baseline(destination=destination, alias=alias)
        raise


def unload_baseline(*, destination: pathlib.Path, alias: str = ALIAS) -> None:
    """Undo the import of a baseline package.

    Remove it from the module cache and from the module search path, as well as its Warp modules (named after its
    Python modules) from Warp's registry, so that they are not loaded (e.g. by a later CUDA graph capture).
    Warp modules named by content (``module="unique"``) are kept, since they can be shared with other packages.

    :param destination: The directory where the package was exported.
    :param alias: The name of the exported package.
    """

    def owned(name: str) -> bool:
        return name == alias or name.startswith(f"{alias}.")

    for name in [name for name in sys.modules if owned(name)]:
        del sys.modules[name]
    # private registry of the Warp modules (available in the supported Warp versions)
    registry = getattr(getattr(getattr(wp, "_src", None), "context", None), "user_modules", None)
    if isinstance(registry, dict):
        for name in [name for name in registry if owned(name)]:
            del registry[name]
    with contextlib.suppress(ValueError):
        sys.path.remove(str(destination))
    importlib.invalidate_caches()
