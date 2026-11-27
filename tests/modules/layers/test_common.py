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

import warp as wp

from warp_nn.modules.layers._common import can_store_final_states


def _zeros(shape):
    return wp.zeros(shape, dtype=wp.float32, device="cpu")


@pytest.mark.parametrize("batch_first", [True, False])
def test_can_store_final_states(capsys, batch_first):
    sequence = _zeros((4, 3, 2) if batch_first else (3, 4, 2))  # batch_size 4, sequence_length 3, features 2
    initial, final = _zeros((4, 2)), _zeros((4, 2))
    kwargs = {"initial_states": (initial,), "final_states": (final,)}
    # disjoint arrays, in both directions
    assert can_store_final_states(sequence, [0, 1, 2], batch_first, **kwargs)
    assert can_store_final_states(sequence, [2, 1, 0], batch_first, **kwargs)
    # no time step
    assert not can_store_final_states(sequence, [], batch_first, **kwargs)


def test_can_store_final_states_overlapping_input(capsys):
    # sequence-first input whose time steps are the final states of the layer's directions (stacked)
    finals = _zeros((2, 4, 2))
    kwargs = {"initial_states": (_zeros((4, 2)),)}
    # the last time step reads the final states to store: not safe
    assert not can_store_final_states(finals, [0, 1], False, final_states=(finals[1],), **kwargs)
    assert not can_store_final_states(finals, [1, 0], False, final_states=(finals[0],), **kwargs)
    # only an earlier time step reads them (already completed by previous kernels): safe
    assert can_store_final_states(finals, [0, 1], False, final_states=(finals[0],), **kwargs)


def test_can_store_final_states_overlapping_batch_first_input(capsys):
    # batch-first input (batch_size 4, sequence_length 2, features 2), whose time steps are strided views
    sequence = _zeros((4, 2, 2))
    kwargs = {"initial_states": (_zeros((4, 2)),)}
    # the last time step reads (a strided view of) the array storing the final states: not safe
    assert not can_store_final_states(sequence, [0, 1], True, final_states=(sequence[:, 1],), **kwargs)
    # the first time step of the first batch element, right before the memory span of the last time step: safe
    # (its shape is chosen only to probe the boundary of the span, which the function does not check)
    assert can_store_final_states(sequence, [0, 1], True, final_states=(sequence[:1, 0],), **kwargs)


def test_can_store_final_states_overlapping_initial_states(capsys):
    sequence = _zeros((1, 4, 2))
    hidden, cell = _zeros((4, 2)), _zeros((4, 2))
    final_hidden = _zeros((4, 2))
    # a single time step reads the initial states: not safe if any of them overlaps any final state
    assert not can_store_final_states(
        sequence, [0], False, initial_states=(hidden, cell), final_states=(final_hidden, cell)
    )
    assert not can_store_final_states(sequence, [0], False, initial_states=(hidden,), final_states=(hidden,))
    # a single time step whose initial states are disjoint from the final states: safe
    assert can_store_final_states(
        sequence, [0], False, initial_states=(hidden, cell), final_states=(final_hidden, _zeros((4, 2)))
    )
    # several time steps only read the initial states at the first one: safe
    sequence = _zeros((3, 4, 2))
    assert can_store_final_states(sequence, [0, 1, 2], False, initial_states=(hidden,), final_states=(hidden,))
