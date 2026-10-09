Kernel Configuration
====================

Warp-NN kernels can be created and launched with configurable parameters, which are set through
:class:`~warp_nn.utils.config.KernelConfig` and the :func:`~warp_nn.utils.config.kernel_config` context manager.

Tile shapes and ``block_dim``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Warp-NN kernels process most data in tiles: each CUDA thread block loads a tile of the input array, applies the
operation, and stores the result. Two settings of :class:`~warp_nn.utils.config.KernelConfig` control this,
and they describe different things:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Setting
      - Meaning
    * - ``tile_*d``
      - **How much data** each block processes: the logical shape of the tile (for example ``(16, 16)`` is
        256 elements). This also sets the launch grid: an array is split into ``ceil(shape / tile)`` tiles
        along each dimension.
    * - ``block_dim``
      - **How many threads** cooperate on that tile (threads per block, at most 1024).

The number of *tile elements* is the number of array values (scalars) in one tile, that is, the product of its
shape: ``(1024,)`` has 1024 elements, ``(8, 32)`` has 8 x 32 = 256, and ``(4, 8, 32)`` has 1024. It is the amount
of data one block reads and writes in a single pass. The tile shape does not have to match ``block_dim``.
The tile's elements are spread across the block's threads, so each thread handles ``tile elements / block_dim``
of them (the *elements per thread*).

.. code-block:: text

    block_dim = 256

    tile of   64 elements  ->  0.25 elements/thread (192 of 256 threads are idle)
    tile of  256 elements  ->  1    elements/thread (every thread has one element)
    tile of 1024 elements  ->  4    elements/thread (every thread loops over 4 elements)

Guidelines
----------

* **Do not use a tile smaller than** ``block_dim``. Threads beyond the tile size have no work and are wasted.
* **Use a** ``block_dim`` **that is a multiple of 32** (the warp size). The default of 256 is a good general choice
  and matches Warp's own default.
* **Aim for a few elements per thread.** The elementwise operations in Warp-NN are memory-bound, so it helps for
  each thread to move about 16 bytes per pass. As a rule of thumb, this is 2 elements for ``float64``,
  4 for ``float32`` and 8 for ``float16``. With ``block_dim=256`` and ``float32``, that is a tile of about
  1024 elements. Values between 1 and 8 elements per thread are usually close to each other in performance.
* **Prefer a wide last dimension for 2D and higher tiles.** The last dimension is contiguous in memory, while
  consecutive tile rows can be far apart. A wider row therefore means fewer, longer contiguous chunks per tile and
  more coalesced accesses. For example, ``(8, 32)`` and ``(16, 16)`` both have 256 elements, but for ``float32``
  the first reads 8 chunks of 128 bytes (one cache line each) and the second 16 chunks of 64 bytes, so
  ``(8, 32)`` is usually slightly better.
* **Make the last dimension a multiple of 16 bytes.** Warp can use faster 128-bit loads when
  ``last_dim * sizeof(dtype)`` is a multiple of 16 bytes (for example, a multiple of 4 for ``float32``). Both
  ``(8, 32)`` and ``(16, 16)`` satisfy this. The loads also need the array itself to be dense and 16-byte aligned,
  with rows of a multiple of 16 bytes; otherwise Warp silently falls back to scalar loads.
* **Do not oversize tiles for small arrays.** The grid is rounded up, so when an array dimension is smaller than the
  tile dimension, part of the tile is out of bounds and wasted work. A tile that is too large can also produce too
  few blocks to keep the GPU busy.

.. note::

    The best values depend on the GPU, the data type and the array sizes. The guidelines above are starting points:
    if performance matters for your workload, benchmark a few tile shapes and ``block_dim`` values.

Default values
--------------

The default configuration uses ``block_dim=256`` and tiles of 1024 elements, that is, 4 elements per thread
for ``float32``:

.. literalinclude:: ../../../warp_nn/utils/config.py
    :language: python
    :start-after: [start-default-config]
    :end-before: [end-default-config]

Example
-------

The configuration is read when a module is created, so create the module inside the context manager:

.. code-block:: python

    from warp_nn import nn
    from warp_nn.utils import kernel_config

    # 1D tile: 1024 elements / 256 threads = 4 elements per thread (16 bytes for float32).
    # E.g., a 1D array of 1M elements is split into `ceil(1_000_000 / 1024) = 977` blocks
    # (the last block is only partially filled).
    with kernel_config(block_dim=256, tile_1d=(1024,)):
        relu = nn.ReLU()

    # Smaller blocks and tiles, e.g. for small arrays:
    # - 1D tile: 256 elements / 128 threads = 2 elements per thread.
    # - 2D tile: 8 x 16 = 128 elements / 128 threads = 1 element per thread.
    #   The last dimension of the 2D tile is 16 x 4 bytes = 64 bytes (a multiple of 16),
    #   so float32 can use 128-bit loads.
    with kernel_config(block_dim=128, tile_1d=(256,), tile_2d=(8, 16)):
        sigmoid = nn.Sigmoid()

API
^^^

.. autoclass:: warp_nn.utils.config.KernelConfig
    :members:
    :show-inheritance:
    :undoc-members:

.. autofunction:: warp_nn.utils.config.kernel_config

.. autofunction:: warp_nn.utils.config.get_kernel_config
