.. _installation:

Installation
============

|warp-nn| supports Python versions 3.10 onwards.

Library Installation
--------------------

From PyPI
^^^^^^^^^

To install **warp-nn** from Python Package Index (PyPI), run the following command:

.. code-block:: bash

    pip install warp-nn

From GitHub repository
^^^^^^^^^^^^^^^^^^^^^^

To install |warp-nn| from its GitHub repository, follow one of the following options:

Editable installation
"""""""""""""""""""""

With the editable installation, it is possible to make changes to the library (e.g., add new features or fix bugs)
and test them without reinstalling. In this mode, a shortcut to the original location of the source code
is registered in the Python environment so that any modifications are available immediately.

To install the library in editable mode, run the following commands:

.. code-block:: bash

    git clone https://github.com/NVIDIA/warp-nn.git
    cd warp-nn
    pip install -e .

Git installation
^^^^^^^^^^^^^^^^

To install from a specific Git branch, e.g., ``main``, run the following command:

.. code-block:: bash

    pip install "warp-nn @ git+https://github.com/NVIDIA/warp-nn.git@main"

Dependencies
------------

|warp-nn| supports Python versions 3.10 onwards.

The following required dependencies will be installed automatically:

* `warp-lang <https://nvidia.github.io/warp>`_ ``>= 1.15.0``

Supported Platforms and Requirements
------------------------------------

Since |warp-nn| is built on top of NVIDIA Warp, the same supported platforms and requirements apply.
Refer to the `Warp's Compatibility & Support`_ documentation for more details.

.. _Warp's Compatibility & Support: https://nvidia.github.io/warp/user_guide/compatibility.html
