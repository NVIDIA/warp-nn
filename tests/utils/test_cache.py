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

import gc
import threading

import warp as wp

from warp_nn import nn
from warp_nn.utils import cache


class Owner:
    pass


@pytest.fixture(autouse=True)
def restore_enabled():
    yield
    cache.set_enabled(True)


def test_acquire(capsys):
    owner, other = Owner(), Owner()
    sub_cache = cache.acquire(owner)
    assert isinstance(sub_cache, cache.Cache)
    # the same owner always gets the same sub-cache, and different owners get different sub-caches
    assert cache.acquire(owner) is sub_cache
    assert cache.acquire(other) is not sub_cache
    assert id(owner) in cache._registry
    # owners must support weak references
    with pytest.raises(TypeError):
        cache.acquire(object())


def test_release_on_garbage_collection(capsys):
    owner = Owner()
    sub_cache = cache.acquire(owner)
    owner_id = id(owner)
    sub_cache.get("key", lambda: "value")
    del owner
    gc.collect()
    # the sub-cache is released when its owner is collected (so a new owner reusing its identity starts empty)
    assert owner_id not in cache._registry


def test_release_during_acquire(capsys, monkeypatch):
    # a garbage collection triggered while acquiring (registry lock held) releases a collected owner without deadlock
    owner = Owner()
    owner.cycle = owner  # only collectable by the cycle collector
    cache.acquire(owner)
    owner_id = id(owner)
    del owner
    init = cache.Cache.__init__

    def collecting_init(self):
        gc.collect()
        init(self)

    monkeypatch.setattr(cache.Cache, "__init__", collecting_init)
    thread = threading.Thread(target=cache.acquire, args=(Owner(),), daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive(), "deadlock"
    assert owner_id not in cache._registry


def test_get(capsys):
    sub_cache = cache.acquire(owner := Owner())
    calls = []

    def factory():
        calls.append(None)
        return object()

    # the factory is only called on cache misses
    value = sub_cache.get("a", factory)
    assert sub_cache.get("a", factory) is value
    assert len(calls) == 1
    assert sub_cache.get("b", factory) is not value
    assert len(calls) == 2
    assert "a" in sub_cache and "b" in sub_cache and "c" not in sub_cache
    assert len(sub_cache) == 2
    assert sub_cache.values() == [value, sub_cache.get("b", factory)]
    # nothing is cached if the factory raises
    with pytest.raises(RuntimeError):
        sub_cache.get("c", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert "c" not in sub_cache
    del owner


def test_pop_and_clear(capsys):
    owner, other = Owner(), Owner()
    sub_cache, other_cache = cache.acquire(owner), cache.acquire(other)
    for c in (sub_cache, other_cache):
        c.get("a", object)
        c.get("b", object)
    # remove a single entry (missing keys are ignored)
    sub_cache.pop("a")
    sub_cache.pop("missing")
    assert "a" not in sub_cache and "b" in sub_cache
    # clear a single sub-cache
    sub_cache.clear()
    assert len(sub_cache) == 0 and len(other_cache) == 2
    # clear the sub-cache of an owner
    sub_cache.get("a", object)
    cache.clear(owner)
    assert len(sub_cache) == 0 and len(other_cache) == 2
    cache.clear(Owner())  # owners without sub-cache are ignored
    # clear all the sub-caches, which stay registered
    sub_cache.get("a", object)
    cache.clear()
    assert len(sub_cache) == 0 and len(other_cache) == 0
    assert cache.acquire(owner) is sub_cache and cache.acquire(other) is other_cache


def test_set_enabled(capsys):
    sub_cache = cache.acquire(owner := Owner())
    value = sub_cache.get("a", object)
    assert cache.is_enabled()
    cache.set_enabled(False)
    assert not cache.is_enabled()
    # while disabled, the factory is called at every invocation and nothing is read from or written to the cache
    assert sub_cache.get("a", object) is not value
    first = sub_cache.get("b", object)
    assert sub_cache.get("b", object) is not first
    assert "b" not in sub_cache
    # data cached before disabling is kept
    cache.set_enabled(True)
    assert sub_cache.get("a", object) is value
    del owner


def test_module_cache(capsys):
    module = nn.ReLU().to("cpu")
    assert module.cache is cache.acquire(module)
    # re-initializing keeps the same sub-cache
    sub_cache = module.cache
    nn.Module.__init__(module)
    assert module.cache is sub_cache
    input = wp.ones((2, 4), dtype=wp.float32, device="cpu")
    output = module(input)
    assert module(input) is output
    assert module.cache.values() == [output]
    # disabling the cache allocates a new output array at every call
    cache.set_enabled(False)
    uncached = module(input)
    assert uncached is not output and module(input) is not uncached
    assert (uncached.numpy() == output.numpy()).all()
    cache.set_enabled(True)
    assert module(input) is output
    # clearing the cache of the module allocates a new output array
    cache.clear(module)
    assert module(input) is not output
