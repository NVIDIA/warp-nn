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

"""Global cache of the data (e.g. the output arrays of the forward pass) owned by modules and other objects.

The global cache is organized in sub-caches, one per owner. An owner acquires its sub-cache (see :py:func:`acquire`)
when it is created, and fetches (or creates, on first access) its cached data through it (see :py:meth:`Cache.get`).
A sub-cache is released (and its data freed) when its owner is garbage-collected.

The cache can be cleared (see :py:func:`clear`), and disabled (see :py:func:`set_enabled`).
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, TypeVar

import threading
import weakref


__all__ = ["Cache", "acquire", "clear", "is_enabled", "set_enabled"]

T = TypeVar("T")

_registry: dict[int, Cache] = {}  # sub-caches, by owner identity
_registry_lock = threading.Lock()  # serializes registrations (releases are lock-free, see _release)
_enabled: bool = True  # process-wide cache state


class Cache:
    def __init__(self) -> None:
        """Sub-cache of the global cache, owned by a single object.

        Instances are created by :py:func:`acquire`, and should not be instantiated directly.
        """
        self._entries: dict[Hashable, Any] = {}

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Fetch the cached data for the given key, or create (and cache) it with the factory if it is not cached.

        If the cache is disabled (see :py:func:`is_enabled`), the cached data is bypassed: the factory is called
        at every invocation, and its result is returned without being cached.

        .. warning::

            When the cache is disabled, new data (e.g. arrays) is created at every invocation,
            so it must not be referenced by CUDA graphs that outlive the invocation.

        :param key: Key of the cached data (e.g. the input shape and data type).
        :param factory: Function (without arguments) that creates the data.

        :return: The cached (or created) data.
        """
        if not _enabled:
            return factory()
        try:
            return self._entries[key]
        except KeyError:
            value = self._entries[key] = factory()
            return value

    def values(self) -> list[Any]:
        """Get the cached data.

        :return: A list with the cached data, in insertion order.
        """
        return list(self._entries.values())

    def pop(self, key: Hashable) -> None:
        """Remove the cached data for the given key, if it is cached.

        :param key: Key of the cached data.
        """
        self._entries.pop(key, None)

    def clear(self) -> None:
        """Remove all the cached data of the sub-cache."""
        self._entries.clear()


def acquire(owner: object) -> Cache:
    """Acquire the sub-cache of an owner, registering it in the global cache on first acquisition.

    The sub-cache is identified by the owner's identity (:py:func:`id`), so successive acquisitions for the same owner
    return the same sub-cache. The sub-cache is released from the global cache (and its cached data freed)
    when the owner is garbage-collected.

    :param owner: Object owning the sub-cache (e.g. a module). It must support weak references.

    :return: The sub-cache of the owner.

    :raises TypeError: If the owner does not support weak references.
    """
    owner_id = id(owner)
    with _registry_lock:
        sub_cache = _registry.get(owner_id)
        if sub_cache is None:
            # an identity is only reused after the object is garbage-collected, by which time its sub-cache
            # has been released by the finalizer (which must not hold a strong reference to the owner)
            finalizer = weakref.finalize(owner, _release, owner_id)
            finalizer.atexit = False  # don't free (e.g. device) data during interpreter shutdown
            sub_cache = _registry[owner_id] = Cache()
    return sub_cache


def _release(owner_id: int) -> None:
    # lock-free (dict.pop is atomic): the finalizer can run during a garbage collection triggered while the
    # registry lock is held by the same thread (e.g. within acquire), so taking the lock here could deadlock
    _registry.pop(owner_id, None)


def clear(owner: object | None = None) -> None:
    """Remove the cached data of an owner, or of all the owners.

    The sub-caches stay registered (i.e. sub-caches already acquired remain valid), only their data is removed.

    .. warning::

        The removed data (e.g. arrays) is freed, so it must not be referenced by CUDA graphs captured before clearing.

    :param owner: Object whose cached data is removed. If ``None``, the cached data of all the owners is removed.
    """
    with _registry_lock:
        sub_caches = list(_registry.values()) if owner is None else [_registry.get(id(owner))]
    for sub_cache in sub_caches:
        if sub_cache is not None:
            sub_cache.clear()


def is_enabled() -> bool:
    """Whether the cache is enabled.

    :return: Whether the cache is enabled.
    """
    return _enabled


def set_enabled(enabled: bool) -> None:
    """Enable or disable the cache process-wide.

    While disabled, the cached data is bypassed (see :py:meth:`Cache.get`). Data already cached is kept.

    :param enabled: Whether to enable (true) or disable (false) the cache.
    """
    global _enabled
    _enabled = bool(enabled)
