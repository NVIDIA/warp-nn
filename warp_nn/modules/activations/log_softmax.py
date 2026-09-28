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

from typing import Any

import warp as wp

from warp_nn.modules.activations._softmax import SoftmaxBase, overload_kernels


@wp.kernel
def _log_softmax_kernel(
    input: wp.array3d[Any], maximum: wp.array3d[Any], total: wp.array3d[Any], output: wp.array3d[Any]
):
    i, j, k = wp.tid()
    m = maximum[i, 0, k]
    output[i, j, k] = output.dtype(m.dtype(input[i, j, k]) - m - wp.log(total[i, 0, k]))


_LOG_SOFTMAX_KERNELS = overload_kernels(_log_softmax_kernel, num_accumulation_arrays=2, has_output=True)


class LogSoftmax(SoftmaxBase):
    def __init__(self, *, dim: int = -1, requires_grad: bool = True) -> None:
        r"""Log-Softmax activation function.

        This class computes the logarithm of the Softmax activation function along the given dimension:

        .. math::

            \text{LogSoftmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)
                = x_i - \log\left(\sum_j e^{x_j}\right)

        :param dim: The dimension along which the Log-Softmax function is computed.
        :param requires_grad: Whether the cached output arrays of the module require gradients.
        """
        super().__init__(dim=dim, output_kernels=_LOG_SOFTMAX_KERNELS, requires_grad=requires_grad)
