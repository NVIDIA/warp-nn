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
* **Use a** ``block_dim`` **that is a multiple of 32**, the *warp size*. Here, a *warp* is a GPU hardware concept,
  unrelated to the NVIDIA Warp library: a group of 32 threads that the GPU runs together, executing the same
  instruction at the same time. A block always runs as whole warps, so a ``block_dim`` of 100 still occupies
  4 warps (128 threads), and 28 of them are idle. The default of 256 is a good general choice and matches the
  Warp library's own default.
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
  few blocks to keep the GPU busy. :ref:`Adaptive tiles <adaptive-tiles>` (enabled by default) take care of this
  automatically.

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

.. _adaptive-tiles:

Adaptive tiles
^^^^^^^^^^^^^^

The configured tiles suit large arrays, but waste work on smaller ones. For example, a ``(16, 64)`` tile over an
array of 8 columns computes 64 columns, so 7/8 of its work is out of bounds. And an array covered by only a few
tiles leaves most of the GPU's SMs (streaming multiprocessors) idle.

When ``adaptive`` is true (the default), the modules adapt their tiles and ``block_dim`` to the shape of each array
they process. On CPU devices, which have no SMs to keep busy, the tiles are adapted without the shrink step (step 3
below) or the one thread per tile element that small grids get in step 4. Warp launches the tiled kernels on CPU
devices with one thread per block, so only the adapted tile matters there. The configured values act as upper bounds:

* An adapted tile never has **more elements** than the configured tile, although a single dimension may grow beyond
  its configured size (e.g. more rows when there are fewer columns).
* The adapted ``block_dim`` is never larger than the configured one.

.. _how-a-tile-is-adapted:

How a tile is adapted
---------------------

The tile is adapted to the launch shape (the array covered by the tiles) in four steps:

1. **Narrow**: each tile dimension steps down (see below) while the smaller size still covers the array dimension.
2. **Grow**: the tile dimensions that do not cover the array are doubled (last first), as long as the tile has no
   more elements than the configured tile.
3. **Shrink**: if the grid still has too few blocks to keep every SM busy (fewer than 2 blocks per SM for
   elementwise operations, 4 for matrix multiplications), the tile dimensions step down (leading ones first),
   down to a tile of 128 elements.
4. **Block dimension**: the block keeps the configured number of elements per thread, with its number of threads
   rounded up to a multiple of the GPU warp size (32). If the grid still has fewer than 2 blocks per SM, it gets
   one thread per tile element instead, up to the configured block dimension. Elementwise operations also do this
   when the tile shrinks in step 3. The number of threads is then
   raised (never lowered) to the next multiple of 32, up to the configured ``block_dim``, that divides the tile
   elements, if any, so that every thread gets the same number of elements.

A dimension steps down to the next smaller size of the form *minimum size × power of two* (the minimum size is 1,
or 16 rows and 8 columns for matrix multiplications). For example, 64 steps down to 32, and 40 steps down to 32,
16 and then 8. The sizes reached by stepping down are therefore multiples of the minimum size, and there are few
of them, so few kernel variants are compiled. Tiles whose dimensions are powers of two are simply halved.

Matrix multiplications (the ``Linear``, ``RNN``, ``GRU`` and ``LSTM`` layers and their cells) have a few more
constraints, given by their tile of ``(batch rows, output features)``:

* The tile keeps at least 16 rows and 8 columns (the shape of the tensor-core instructions).
* The rows plus columns, which set the shared memory used by the operand tiles, are limited to those of the
  configured tile.
* When a Linear inference grid reaches the block target, the block dimension is capped at one warp per 32 tile columns.
* The tile does not grow (step 2) for launches recorded on a :class:`warp.Tape` by a module that requires
  gradients, since the backward kernels run on the same tile and were measured to be fastest with 16 rows.

Examples
--------

With the default configuration (``tile_1d=(1024,)``, ``tile_2d=(16, 64)`` and ``block_dim=256``), on a GPU with
76 SMs (the launch shape of a matrix multiplication is its output, ``(batch rows, output features)``):

.. list-table::
    :header-rows: 1
    :widths: 30 30 20 20

    * - Operation
      - Launch shape
      - Adapted tile
      - Adapted ``block_dim``
    * - Elementwise
      - ``(1_000_000,)``
      - ``(1024,)``
      - 256
    * - Elementwise
      - ``(50_000,)``
      - ``(256,)``
      - 256
    * - Elementwise
      - ``(300,)``
      - ``(128,)``
      - 128
    * - Elementwise
      - ``(4096, 8)``
      - ``(16, 8)``
      - 128
    * - Linear inference
      - ``(65536, 512)``
      - ``(16, 64)``
      - 64
    * - Matrix multiplication
      - ``(65536, 8)``
      - ``(64, 8)``
      - 128
    * - Linear inference
      - ``(65536, 8)``
      - ``(64, 8)``
      - 32
    * - Matrix multiplication (recorded for differentiation)
      - ``(65536, 8)``
      - ``(16, 8)``
      - 32

For example, the ``(50_000,)`` array would only get 49 blocks of the configured ``(1024,)`` tile, fewer than the
152 needed to give 2 blocks to each of the 76 SMs. The tile is halved twice to ``(256,)`` (196 blocks).
Since the tile shrank, the block uses one thread per element (256 threads). The ``(300,)`` array gets only 3 blocks
with the smallest tile of 128 elements, so it also gets one thread per element.

Compilation and caching
-----------------------

Within a module, each adapted configuration gets its own kernels, which are compiled the first time an array needs
them. Arrays whose shapes are adapted to the same configuration share the same kernels, and the adapted configuration
of each shape is cached, so that later launches with that shape only do a lookup. If a workload uses many different
array shapes, this may compile more kernels; set ``adaptive=False`` to always use the configured tiles. Disabling it
is also useful to benchmark specific tile shapes, since they are then used as they are:

.. code-block:: python

    from warp_nn import nn
    from warp_nn.utils import kernel_config

    # use the configured tile and block dimension for every array, whatever its shape
    with kernel_config(tile_1d=(512,), block_dim=128, adaptive=False):
        relu = nn.ReLU()

Custom kernels
--------------

Kernels written outside Warp-NN can be adapted the same way, with :py:meth:`~warp_nn.utils.config.KernelConfig.adapt`.
It returns the configuration adapted to a launch shape on a device (or the configuration itself, if nothing changes),
under the adaptation policy of the operation (:py:class:`~warp_nn.utils.config.AdaptationPolicy`). The adapted
configurations are cached per configuration instance, so keep the configuration rather than getting a new one for
each launch:

.. code-block:: python

    import warp as wp
    from warp_nn.utils import AdaptationPolicy, get_kernel_config

    config = get_kernel_config()
    adapted = config.adapt((4096, 8), wp.get_device("cuda:0"), policy=AdaptationPolicy.MATMUL)
    # compile (once per adapted configuration) and launch the kernel with adapted.tile_2d and adapted.block_dim

API
^^^

.. autoclass:: warp_nn.utils.config.KernelConfig
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass:: warp_nn.utils.config.AdaptationPolicy
    :members:
    :show-inheritance:
    :undoc-members:

.. autofunction:: warp_nn.utils.config.kernel_config

.. autofunction:: warp_nn.utils.config.get_kernel_config
