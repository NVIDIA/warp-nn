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

import warp as wp

from warp_nn.modules.module import Module


class Identity(Module):
    def __init__(self, *args, **kwargs) -> None:
        r"""Identity operation.

        This class is placeholder identity operator that is argument-insensitive:

        .. math::

            \text{Identity}(x) = x

        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(requires_grad=kwargs.get("requires_grad", True))

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the operation.

        :param input: The input array, with up to 3 dimensions.

        :return: The output array, with same shape as the input array.
        """
        return input
