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

import dataclasses
from functools import partial

import numpy as np
import warp as wp

from warp_nn import nn
from warp_nn.utils import AdaptationPolicy, cache, get_kernel_config, kernel_config
from warp_nn.utils.config import _adapt_tile


NUM_SMS = 76  # matrix multiplications target 4 * 76 = 304 blocks, and one thread per element below 2 * 76 = 152 blocks


@pytest.mark.parametrize(
    "shape, expected",
    [
        # narrowed to 8 columns, rows grown up to the configured sum of dimensions (64 + 8 <= 16 + 64),
        # enough blocks: the configured 4 elements per thread
        ((65536, 8), ((64, 8), 128)),
        # rows shrunk to their minimum, below the target (256 < 304 blocks) but above the one-thread-per-element
        # threshold (256 >= 152 blocks): the configured 4 elements per thread
        ((4096, 8), ((16, 8), 32)),
        # columns shrunk (after the rows) until there are enough blocks: the configured 4 elements per thread
        ((1024, 64), ((16, 8), 32)),
        # minimum tile with too few blocks: one thread per element
        ((64, 128), ((16, 8), 128)),
        # large array: the configured tile and block dimension
        ((65536, 256), ((16, 64), 256)),
        # empty array: narrowed, but not shrunk (no blocks to provide)
        ((0, 8), ((16, 8), 32)),
    ],
)
def test_adapt_tile_matmul(capsys, shape, expected):
    adapted = _adapt_tile((16, 64), 256, shape, NUM_SMS, policy=AdaptationPolicy.MATMUL)
    assert adapted == expected
    # the adapted tile never has more elements than the configured tile
    assert adapted[0][0] * adapted[0][1] <= 16 * 64


@pytest.mark.parametrize(
    "shape, expected",
    [
        # large array: the configured tile and block dimension
        ((1_000_000,), ((1024,), 256)),
        # shrunk down to the minimum number of elements with too few blocks: one thread per element
        ((1000,), ((128,), 128)),
        # shrunk to 196 blocks, enough for the grid target, and still uses one thread per element
        ((50000,), ((256,), 256)),
        # 2D: narrowed to 8 columns, rows grown to keep the configured elements and 4 elements per thread
        ((65536, 8), ((128, 8), 256)),
    ],
)
def test_adapt_tile_elementwise(capsys, shape, expected):
    tile = (1024,) if len(shape) == 1 else (16, 64)
    assert _adapt_tile(tile, 256, shape, NUM_SMS, policy=AdaptationPolicy.ELEMENTWISE) == expected


@pytest.mark.parametrize(
    "shape, expected",
    [
        # not grown: the configured rows (16), with enough blocks for the configured 4 elements per thread
        ((65536, 8), ((16, 8), 32)),
        ((65536, 32), ((16, 32), 128)),
        # not grown nor shrunk: as without gradients
        ((65536, 256), ((16, 64), 256)),
        ((1024, 64), ((16, 8), 32)),
        ((64, 128), ((16, 8), 128)),
    ],
)
def test_adapt_tile_matmul_backward(capsys, shape, expected):
    assert _adapt_tile((16, 64), 256, shape, NUM_SMS, policy=AdaptationPolicy.MATMUL, backward=True) == expected


@pytest.mark.parametrize(
    "shape, backward, expected",
    [
        ((65536, 8), False, ((64, 8), 32)),
        ((65536, 64), False, ((16, 64), 64)),
        ((64, 8), False, ((16, 8), 128)),
        ((65536, 64), True, ((16, 64), 256)),
    ],
)
def test_adapt_tile_linear(shape, backward, expected):
    assert _adapt_tile((16, 64), 256, shape, NUM_SMS, policy=AdaptationPolicy.LINEAR, backward=backward) == expected


@pytest.mark.parametrize(
    "tile, shape, policy, expected",
    [
        # columns narrowed down the ladder of 8 * 2**k (40 -> 32 -> 16 -> 8), then rows grown up to the configured
        # sum of dimensions (32 + 8 <= 16 + 40)
        ((16, 40), (65536, 4), AdaptationPolicy.MATMUL, ((32, 8), 128)),
        # 48 -> 32 -> 16 -> 8 columns, rows grown up to the configured sum of dimensions (32 + 8 <= 16 + 48); the
        # configured 3 elements per thread give 86 threads (96 for whole warps), raised to 128 to divide the 256 elements
        ((16, 48), (65536, 4), AdaptationPolicy.MATMUL, ((32, 8), 128)),
        # narrowed to the smallest ladder size covering the array (24 -> 16 covers 12 columns, 8 does not)
        ((16, 24), (65536, 12), AdaptationPolicy.MATMUL, ((16, 16), 256)),
        # columns that cover the array are kept (32 does not cover 40); the configured 2 elements per thread give
        # 320 threads, capped at 256, which does not divide the 640 elements (the threads are never lowered)
        ((16, 40), (65536, 40), AdaptationPolicy.MATMUL, ((16, 40), 256)),
        # shrunk down the ladder for occupancy (24 -> 16 -> 8 columns), as a power of two would be
        ((16, 24), (1024, 64), AdaptationPolicy.MATMUL, ((16, 8), 128)),
        # rows of 24 (above the minimum of 16) are shrunk to 16, not kept for not being a multiple of 32
        ((24, 64), (1024, 64), AdaptationPolicy.MATMUL, ((16, 8), 32)),
        # element-wise: 48 -> 32 -> 16 -> 8 (the ladder of 1 * 2**k), 4 does not cover 5 elements
        ((48,), (5,), AdaptationPolicy.ELEMENTWISE, ((8,), 32)),
        # a configured size below its minimum size is its own minimum size (not narrowed)
        ((16, 6), (65536, 2), AdaptationPolicy.MATMUL, ((16, 6), 96)),
    ],
)
def test_adapt_tile_non_power_of_two(capsys, tile, shape, policy, expected):
    adapted = _adapt_tile(tile, 256, shape, NUM_SMS, policy=policy)
    assert adapted == expected
    # the adapted dimensions are on the ladder of their minimum size (minimum * 2**k), or the configured size (none
    # of these cases grows a size off the ladder, e.g. 24 rows doubled to 48)
    for size, configured, low in zip(adapted[0], tile, policy.min_tile(len(tile))):
        minimum = min(low, configured)
        ratio = size // minimum
        assert size == configured or (size % minimum == 0 and ratio & (ratio - 1) == 0)


@pytest.mark.parametrize("num_sms", [1, 16, 76, 142])
def test_adapt_tile_power_of_two_unchanged_by_ladder(capsys, num_sms):
    # for powers of two, stepping down the ladder of minimum * 2**k is halving: the adapted tiles stay powers of two,
    # and their block dimension divides their elements (every thread gets the same number of elements)
    for tile in ((16, 64), (32, 32), (16, 16)):
        for batch_size in (1, 7, 64, 1000, 4096, 65536):
            for features in (1, 3, 8, 12, 64, 100, 256):
                (rows, columns), block_dim = _adapt_tile(
                    tile, 256, (batch_size, features), num_sms, policy=AdaptationPolicy.MATMUL
                )
                assert rows & (rows - 1) == 0 and columns & (columns - 1) == 0
                assert (rows * columns) % block_dim == 0


def test_adapt_tile_elementwise_backward(capsys):
    # the element-wise tiles are grown whether or not their backward kernels run
    for backward in (False, True):
        adapted = _adapt_tile(
            (16, 64), 256, (65536, 8), NUM_SMS, policy=AdaptationPolicy.ELEMENTWISE, backward=backward
        )
        assert adapted == ((128, 8), 256)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("num_sms", [1, 16, 76, 142])
def test_adapt_tile_matmul_bounds(capsys, num_sms, backward):
    # the adapted tile never needs more shared memory than the configured tile: no more elements (output tile), no
    # larger sum of dimensions (operand tiles), and the block dimension is at most the configured one
    for batch_size in (0, 1, 7, 64, 1000, 4096, 65536):
        for features in (1, 3, 8, 12, 64, 100, 256):
            (rows, columns), block_dim = _adapt_tile(
                (16, 64), 256, (batch_size, features), num_sms, policy=AdaptationPolicy.MATMUL, backward=backward
            )
            assert rows * columns <= 16 * 64 and rows + columns <= 16 + 64
            assert rows >= 16 and columns >= 8 and 32 <= block_dim <= 256


def test_adapt_tile_keeps_non_positive_block_dim(capsys):
    assert _adapt_tile((16, 64), 0, (64, 128), NUM_SMS, policy=AdaptationPolicy.MATMUL) == ((16, 8), 0)
    assert _adapt_tile((16, 64), -1, (65536, 8), NUM_SMS, policy=AdaptationPolicy.MATMUL) == ((64, 8), -1)


def test_adapt_tile_errors(capsys):
    with pytest.raises(ValueError, match="positive"):
        _adapt_tile((0, 64), 256, (16, 64), NUM_SMS, policy=AdaptationPolicy.ELEMENTWISE)
    with pytest.raises(ValueError, match="same number of dimensions"):
        _adapt_tile((16, 64), 256, (16,), NUM_SMS, policy=AdaptationPolicy.ELEMENTWISE)


def test_adapt_unchanged(capsys):
    config = get_kernel_config()
    with kernel_config(adaptive=False):
        disabled = get_kernel_config()
    for device in ("cpu", *(["cuda:0"] if wp.is_cuda_available() else [])):
        # no tile with the launch shape's number of dimensions
        assert config.adapt((1, 2, 3, 4, 5), wp.get_device(device)) is config
        # disabled
        assert disabled.adapt((64, 8), wp.get_device(device), policy=AdaptationPolicy.MATMUL) is disabled
        # adapting yields the configured values (a large array)
        assert config.adapt((65536, 1024), wp.get_device(device)) is config


def test_adapt_cpu(capsys):
    device = wp.get_device("cpu")
    with kernel_config(tile_1d=(1024,), tile_2d=(16, 64), block_dim=256, adaptive=True):
        config = get_kernel_config()
    # adapted without SMs (no shrink step, and no one thread per element for small grids)
    for shape, policy in [
        ((64, 8), AdaptationPolicy.MATMUL),
        ((64, 256), AdaptationPolicy.LINEAR),
        ((300,), AdaptationPolicy.ELEMENTWISE),
    ]:
        adapted = config.adapt(shape, device, policy=policy)
        tile = getattr(config, f"tile_{len(shape)}d")
        expected = _adapt_tile(tile, config.block_dim, shape, 0, policy=policy)
        assert (getattr(adapted, f"tile_{len(shape)}d"), adapted.block_dim) == expected
    # narrowed (and grown), but not shrunk: the narrow (64, 8) output gets 64 rows of 8 columns, while the
    # (64, 256) output keeps the configured tile, which a CUDA device would shrink for more blocks
    assert config.adapt((64, 8), device, policy=AdaptationPolicy.MATMUL).tile_2d == (64, 8)
    assert config.adapt((64, 256), device, policy=AdaptationPolicy.LINEAR) is config
    assert config.adapt((300,), device).tile_1d == (512,)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_adapt_cuda(capsys):
    device = wp.get_device("cuda:0")
    config = get_kernel_config()
    adapted = config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL)
    tile, block_dim = _adapt_tile(
        config.tile_2d, config.block_dim, (65536, 8), device.sm_count, policy=AdaptationPolicy.MATMUL
    )
    assert (adapted.tile_2d, adapted.block_dim) == (tile, block_dim)
    # adapting a known launch shape is a lookup (the same adapted configuration)
    assert config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL) is adapted
    # an adaptation that yields the configured tile and block dimension reuses the configuration
    assert config.adapt((65536, 256), device, policy=AdaptationPolicy.MATMUL) is config
    # the other tiles are kept
    assert (adapted.tile_1d, adapted.tile_3d, adapted.tile_4d) == (config.tile_1d, config.tile_3d, config.tile_4d)
    # the policies are distinguished (the element-wise policy grows the rows of the (65536, 8) tile beyond 64)
    elementwise = config.adapt((65536, 8), device)
    assert elementwise == config.adapt((65536, 8), device, policy=AdaptationPolicy.ELEMENTWISE) != adapted
    # equal configurations, created independently, adapt equally
    assert get_kernel_config().adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL) == adapted


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_adapt_backward(capsys):
    device = wp.get_device("cuda:0")
    with kernel_config(tile_2d=(16, 64), block_dim=256, adaptive=True):
        config = get_kernel_config()
    forward = config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL)
    # the backward kernels are only considered if gradients are required and a tape records the launch
    assert config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL, requires_grad=True) is forward
    with wp.Tape():
        assert config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL) is forward
        backward = config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL, requires_grad=True)
    tile, block_dim = _adapt_tile(
        config.tile_2d, config.block_dim, (65536, 8), device.sm_count, policy=AdaptationPolicy.MATMUL, backward=True
    )
    assert (backward.tile_2d, backward.block_dim) == (tile, block_dim)
    # policies that grow their tiles for the backward kernels too ignore it
    with wp.Tape():
        assert config.adapt((65536, 8), device, requires_grad=True) is config.adapt((65536, 8), device)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_adapt_equal_policies(capsys):
    device = wp.get_device("cuda:0")
    config = get_kernel_config()
    # policies are compared by value: equal policies, created for each launch, share the adapted configuration
    policies = [dataclasses.replace(AdaptationPolicy.MATMUL, blocks_per_sm=8) for _ in range(3)]
    adapted = [config.adapt((65536, 8), device, policy=policy) for policy in policies]
    assert all(other is adapted[0] for other in adapted) and len(config._adapted) == 1
    assert dataclasses.replace(AdaptationPolicy.LINEAR) == AdaptationPolicy.LINEAR != AdaptationPolicy.MATMUL


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_adapt_errors(capsys):
    with kernel_config(tile_2d=(0, 64)):
        config = get_kernel_config()
    with pytest.raises(ValueError, match="must be positive"):
        config.adapt((64, 8), wp.get_device("cuda:0"))


def test_kernel_config_hash(capsys):
    config = get_kernel_config()
    with kernel_config(tile_2d=(16, 32)):
        other = get_kernel_config()
    # equal configurations hash equally (the hash is cached per configuration)
    assert hash(config) == hash(config) == hash(get_kernel_config()) and config == get_kernel_config()
    assert config != other and len({config, get_kernel_config(), other}) == 2
    # the adapted configurations do not take part in the equality
    config.adapt((64, 8), wp.get_device("cpu"))
    assert config == get_kernel_config() and "_adapted" not in repr(config)


class _Module(nn.Module):
    pass


def test_module_kernels(capsys):
    calls = []

    def create_kernels(*, config, value):
        calls.append(config)
        return (config, value)

    config = get_kernel_config()
    with kernel_config(tile_2d=(16, 32)):
        other = get_kernel_config()
    module = _Module()
    kernels = module.kernels
    assert module.kernels is kernels and kernels.factory is None and len(kernels) == 0
    assert repr(kernels) == "Kernels(factory=None, configs=0)"
    # the kernel factory must be set before fetching the kernels, and it must be callable
    with pytest.raises(RuntimeError, match="The kernel factory has not been set"):
        kernels.select(config=config)
    with pytest.raises(TypeError, match="kernel factory must be callable, got dict"):
        kernels.factory = {}
    factory = partial(create_kernels, value=1)
    kernels.factory = factory
    assert kernels.factory is factory
    # the kernels are created the first time a configuration is requested, with the arguments bound to the factory,
    # and then looked up
    assert config not in kernels
    assert kernels.select(config=config) == (config, 1)
    assert config in kernels and len(kernels) == 1
    assert kernels.select(config=config) == (config, 1)
    # equal configurations share the kernels, and other configurations get their own
    assert kernels.select(config=get_kernel_config()) is kernels.select(config=config)
    assert kernels.select(config=other) == (other, 1)
    assert calls == [config, other] and len(kernels) == 2
    assert repr(kernels) == f"Kernels(factory={factory!r}, configs=2)"
    # the kernels are kept when the cache is cleared or disabled, and when the module is moved to another device
    kernel = kernels.select(config=config)
    cache.clear(module)
    module.to("cpu")
    cache.set_enabled(False)
    try:
        assert kernels.select(config=config) is kernel
    finally:
        cache.set_enabled(True)
    assert calls == [config, other]
    # setting another factory discards the kernels created by the previous one
    kernels.factory = partial(create_kernels, value=2)
    assert len(kernels) == 0
    assert kernels.select(config=config) == (config, 2)
    assert calls == [config, other, config]
    # the key selects the kernel from the collection created by the factory (e.g. overloads by ndim and dtype)
    kernels.factory = lambda *, config: {(2, wp.float32): ("2d", config), (3, wp.float32): ("3d", config)}
    assert kernels.select(config=config, key=(2, wp.float32)) == ("2d", config)
    assert kernels.select(config=config, key=(3, wp.float32)) == ("3d", config)
    assert len(kernels) == 1
    with pytest.raises(KeyError):
        kernels.select(config=config, key=(1, wp.float32))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_module_kernels_cuda(capsys):
    calls = []

    def create_kernels(*, config):
        calls.append(config)
        return config

    with kernel_config(tile_2d=(16, 64), block_dim=256, adaptive=True):
        config = get_kernel_config()
    kernels = _Module().kernels
    kernels.factory = create_kernels
    device = wp.get_device("cuda:0")
    # launch shapes with different adapted configurations get their own kernels
    narrow = kernels.select(config=config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL))
    assert narrow == config.adapt((65536, 8), device, policy=AdaptationPolicy.MATMUL) != config
    assert kernels.select(config=config.adapt((65536, 256), device, policy=AdaptationPolicy.MATMUL)) is config
    # launch shapes with the same adapted configuration share them
    assert kernels.select(config=config.adapt((131072, 8), device, policy=AdaptationPolicy.MATMUL)) is narrow
    # (configurations adapted on CPU devices too, without SMs, share them if equal)
    cpu = config.adapt((65536, 8), wp.get_device("cpu"), policy=AdaptationPolicy.MATMUL)
    assert cpu == narrow and kernels.select(config=cpu) is narrow
    assert calls == [narrow, config]


def _layer(name: str, requires_grad: bool, rng: np.random.Generator, device: wp.Device) -> tuple[nn.Module, tuple]:
    # a layer (with 4 input features and 8 output features) and its call on a batch of 65536 samples
    batch_size, cell = 65536, name.endswith("Cell")
    module = getattr(nn, name)(4, 8, requires_grad=requires_grad).to(device)

    def array(*shape):
        return wp.array(rng.uniform(-1, 1, shape).astype(np.float32), device=device, requires_grad=requires_grad)

    if name == "Linear":
        inputs = (array(batch_size, 4),)
    elif name == "LSTMCell":
        inputs = (array(batch_size, 4), (array(batch_size, 8), array(batch_size, 8)))
    elif cell:
        inputs = (array(batch_size, 4), array(batch_size, 8))
    else:  # sequence of 3 time steps
        inputs = (array(batch_size, 3, 4),)
    return module, inputs


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("layer", ["Linear", "RNN", "GRU", "LSTM", "RNNCell", "GRUCell", "LSTMCell"])
def test_layer_tile_backward(capsys, layer):
    # the layers adapt their tiles for the backward kernels too only if they require gradients and a tape records them
    device = wp.get_device("cuda:0")
    with kernel_config(tile_2d=(16, 64), block_dim=256, adaptive=True):
        config = get_kernel_config()
        modules = {}
        for requires_grad in (False, True):
            modules[requires_grad] = _layer(layer, requires_grad, np.random.default_rng(0), device)
    modules[True][0].load_state_dict(modules[False][0].state_dict())
    policy = AdaptationPolicy.LINEAR if layer == "Linear" else AdaptationPolicy.MATMUL
    expected = {
        backward: _adapt_tile(
            config.tile_2d,
            config.block_dim,
            (65536, 8),
            device.sm_count,
            policy=policy,
            backward=backward,
        )
        for backward in (False, True)
    }

    def adapted_tile(module):
        adapted = module.config.adapt((65536, 8), device, policy=policy, requires_grad=module.requires_grad)
        return adapted.tile_2d, adapted.block_dim

    def flat(outputs):
        if isinstance(outputs, (tuple, list)):
            return np.concatenate([flat(output) for output in outputs])
        return outputs.numpy().ravel()

    outputs = {}
    for requires_grad, (module, inputs) in modules.items():
        outputs[requires_grad] = flat(module(*inputs))  # not recorded: the same tile
    with wp.Tape():
        recorded = {requires_grad: flat(module(*inputs)) for requires_grad, (module, inputs) in modules.items()}
    # the launches adapted their tiles to the (65536, 8) output (and only the module that requires gradients for the
    # backward pass)
    assert set(modules[False][0].config._adapted) == {((65536, 8), device.alias, policy, False)}
    assert set(modules[True][0].config._adapted) == {
        ((65536, 8), device.alias, policy, backward) for backward in (False, True)
    }
    assert adapted_tile(modules[False][0]) == expected[False]
    with wp.Tape():
        assert adapted_tile(modules[True][0]) == expected[True]
    # the same outputs with every tile
    for result in (outputs[True], recorded[False], recorded[True]):
        np.testing.assert_allclose(result, outputs[False], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "policy, ndim, expected",
    [
        # the minimum sizes apply to the last dimensions, and the leading ones have a minimum size of 1
        (AdaptationPolicy.MATMUL, 2, [16, 8]),
        (AdaptationPolicy.MATMUL, 3, [1, 16, 8]),
        # tiles with fewer dimensions than the minimum sizes ignore the leading minimum sizes
        (AdaptationPolicy.MATMUL, 1, [8]),
        # policies without minimum sizes
        (AdaptationPolicy.ELEMENTWISE, 1, [1]),
        (AdaptationPolicy.ELEMENTWISE, 3, [1, 1, 1]),
    ],
)
def test_adaptation_policy_min_tile(capsys, policy, ndim, expected):
    assert policy.min_tile(ndim) == expected
