Cache
=====

Modules cache the data they allocate during the forward pass (e.g. the output arrays, per input shape and data type),
so that it is allocated only once and reused (e.g. by CUDA graphs) in subsequent calls.

The cached data lives in a global cache, organized in sub-caches (one per module). A module acquires its sub-cache
when it is created, which is exposed via the module's ``cache`` property. A sub-cache is released (and its data freed)
when the module is garbage-collected, and it is cleared when the module is moved to another device.

Example
-------

.. code-block:: python

    import warp as wp
    from warp_nn import nn
    from warp_nn.utils import cache

    relu = nn.ReLU()
    input = wp.ones((2, 4), dtype=wp.float32)
    output = relu(input)
    assert relu(input) is output  # the output array is cached

    # modules fetch (or create, on first access) their cached data through their sub-cache
    buffer = relu.cache.get(("buffer", input.shape), lambda: wp.empty_like(input))
    assert relu.cache.get(("buffer", input.shape), lambda: wp.empty_like(input)) is buffer

    # disable the cache process-wide (the data is created at every call)
    cache.set_enabled(False)
    assert relu(input) is not output
    cache.set_enabled(True)

    # remove the cached data of a module, or of all the modules
    cache.clear(relu)
    cache.clear()

.. warning::

    CUDA graphs reference the cached arrays by address. Arrays allocated while the cache is disabled (at every call),
    and arrays removed by clearing the cache (or by moving a module to another device), are freed, so they must not
    be referenced by CUDA graphs that are replayed afterwards.

API
^^^

.. autoclass:: warp_nn.utils.cache.Cache
    :members:
    :special-members: __contains__, __len__

.. autofunction:: warp_nn.utils.cache.acquire

.. autofunction:: warp_nn.utils.cache.clear

.. autofunction:: warp_nn.utils.cache.set_enabled

.. autofunction:: warp_nn.utils.cache.is_enabled
