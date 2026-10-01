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

import inspect
import logging
import operator
import os
import sys

# --------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
print("[DOCS] warp-nn path: {}".format(sys.path[0]))
import warp_nn

# Determine the Git version/tag from CI environment variables.
# 1. Check for custom variable.
# 2. Check for GitHub Actions' variable (GITHUB_* variables are set by GitHub and cannot be overwritten).
# 3. Check for GitLab CI's variable.
# 4. Fallback to 'main' for local builds.
git_reference = (
    os.environ.get("GIT_REF_NAME")
    or os.environ.get("GITHUB_REF_NAME")
    or os.environ.get("CI_COMMIT_REF_NAME")
    or "main"
)

# --------------------------------------------------------------------
# Project information
# --------------------------------------------------------------------

project = "warp-nn"
version = warp_nn.__version__
release = version
copyright = "2026, NVIDIA"
author = "NVIDIA"

# --------------------------------------------------------------------
# General configuration
# --------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    # third-party extensions
    "myst_parser",
    "sphinx_copybutton",
]

# Generate links to the documentation of objects in external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "warp": ("https://nvidia.github.io/warp/stable", None),
}

pygments_style = "tango"
pygments_dark_style = "zenburn"

exclude_patterns = ["README.md"]

intersphinx_disabled_domains = ["std"]
templates_path = ["_templates"]
rst_prolog = """

.. |_1| unicode:: 0xA0
    :trim:

.. |_2| unicode:: 0xA0 0xA0
    :trim:

.. |_3| unicode:: 0xA0 0xA0 0xA0
    :trim:

.. |_4| unicode:: 0xA0 0xA0 0xA0 0xA0
    :trim:

.. |_5| unicode:: 0xA0 0xA0 0xA0 0xA0 0xA0
    :trim:

.. |br| raw:: html

            <br>

.. |hr| raw:: html

            <hr>

.. |nvidia_warp| replace:: `NVIDIA Warp <https://nvidia.github.io/warp/>`__
.. |warp| replace:: `Warp <https://nvidia.github.io/warp/>`__
.. |warp-nn| replace:: Warp-NN

"""

# autodoc ext
autodoc_member_order = "groupwise"
autoclass_content = "init"
autodoc_mock_imports = []

# doctest ext
# the docstring examples (plain doctest blocks) depend on test fixtures, so they are run by pytest instead
doctest_test_doctest_blocks = ""

# copybutton ext
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# mathjax
mathjax3_config = {
    "options": {
        "enableMenu": False,
    }
}


# linkcode ext
def linkcode_resolve(domain, info):
    """Try to generate external links to code hosted on the Warp-NN GitHub."""
    if domain != "py":
        return None
    if not info["module"] or not info["fullname"]:
        return None
    # resolve sources
    try:
        obj = operator.attrgetter(info["fullname"])(sys.modules.get(info["module"]))
        if isinstance(obj, property):
            obj = obj.fget
        filename = inspect.getsourcefile(obj)
        source, linenum = inspect.getsourcelines(obj)
    except Exception:
        return None
    # build URL
    filename = os.path.relpath(filename, start=os.path.dirname(warp_nn.__file__))
    lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""
    return f"https://github.com/NVIDIA/warp-nn/blob/{git_reference}/warp_nn/{filename}{lines}"


# --------------------------------------------------------------------
# HTML output
# --------------------------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_title = f"Warp-NN {version}"
html_show_sphinx = False
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = [
    "css/nvidia-sphinx-theme.css",
    "css/custom.css",
]
html_context = {
    "github_user": "NVIDIA",
    "github_repo": "warp-nn",
    "github_version": git_reference,
    "doc_path": "docs",
}
html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "github_url": "https://github.com/NVIDIA/warp-nn",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/warp-nn",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "content",
    "navbar_start": [
        "navbar-logo",
        # "version-switcher",
    ],
    "switcher": {
        "json_url": "https://nvidia.github.io/warp-nn/switcher.json",
        "version_match": git_reference,
    },
    "show_nav_level": 2,
}

if html_theme == "pydata_sphinx_theme":
    html_theme_options["logo"] = {
        "text": "Warp-NN",
        "image_light": "_static/nvidia-logo-horiz-rgb-blk-for-screen.svg",
        "image_dark": "_static/nvidia-logo-horiz-rgb-wht-for-screen.svg",
    }
    html_theme_options["navbar_center"] = ["navbar-nav"]


class _IntersphinxUnreachableFilter(logging.Filter):
    """Downgrade the "inventory unreachable" warning to INFO so a flaky external site doesn't fail the -W build.

    Sphinx emits this warning without a subtype, so it can't be silenced via ``suppress_warnings``.
    """

    def filter(self, record):
        if record.levelno == logging.WARNING and "failed to reach any of the inventories" in record.getMessage():
            record.levelno, record.levelname = logging.INFO, "INFO"
        return True


logging.getLogger("sphinx.sphinx.ext.intersphinx").addFilter(_IntersphinxUnreachableFilter())
