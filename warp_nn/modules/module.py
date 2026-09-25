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

from __future__ import annotations

from typing import Any

from abc import ABC
from collections import OrderedDict

import numpy as np
import warp as wp

from warp_nn.modules.buffer import Buffer
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import parse_device


class Module(ABC):
    def __init__(self, *args, requires_grad: bool = True, **kwargs):
        """Base abstract class for all the modules.

        Modules can contain other modules (sub-modules), organized in a nested tree structure.
        Such sub-modules can be assigned as regular attributes to the parent module.

        .. important::

            Sub-modules assigned as regular attributes to the parent module are not registered automatically.
            Therefore, it is necessary to call the :py:meth:`__post_init__` method before exiting the
            initialization of the module (i.e. at the end of the class constructor).

        :param args: Additional positional arguments.
        :param requires_grad: Whether the parameters and the cached output arrays of the module require gradients.
        :param kwargs: Additional keyword arguments.
        """
        self._requires_grad: bool = requires_grad
        if not hasattr(self, "_device"):
            self._device: wp.Device = parse_device(None)
        if not hasattr(self, "_training"):
            self._training: bool = True
        if not hasattr(self, "_modules"):
            self._modules: OrderedDict[str, Module] = OrderedDict()
        if not hasattr(self, "_parameters"):
            self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        if not hasattr(self, "_buffers"):
            self._buffers: OrderedDict[str, Buffer] = OrderedDict()

    def __post_init__(self) -> None:
        """Register sub-modules, parameters and buffers.

        .. important::

            A module subclass must call this method to register sub-modules, parameters and buffers assigned as
            regular attributes to it, unless they have already been registered manually.
        """
        # register modules, parameters and buffers
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                self.register_module(k, v)
            elif isinstance(v, Parameter):
                self.register_parameter(k, v)
            elif isinstance(v, Buffer):
                self.register_buffer(k, v)

    def __call__(self, *args, **kwargs) -> Any:
        """Forward pass of the module.

        :raises NotImplementedError: If the module subclass does not implement the method.
        """
        raise NotImplementedError(f"Module '{type(self).__name__}' is missing the required the .__call__(...) method")

    @property
    def device(self) -> wp.Device:
        """Device on which the module is allocated."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Whether the parameters and the cached output arrays of the module require gradients."""
        return self._requires_grad

    @property
    def training(self) -> bool:
        """Whether the module is in training mode (true) or in evaluation mode (false)."""
        return self._training

    def train(self, mode: bool = True) -> Module:
        """Set the module and its registered sub-modules in training or evaluation mode.

        The mode only affects modules that behave differently during training and evaluation
        (e.g. :py:class:`~warp_nn.modules.layers.Dropout` or :py:class:`~warp_nn.modules.layers.BatchNorm`).
        Modules are in training mode by default.

        :param mode: Whether to set training mode (true) or evaluation mode (false).

        :return: The module itself.
        """
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module and its registered sub-modules in evaluation mode.

        This is equivalent to ``module.train(False)``.

        :return: The module itself.
        """
        return self.train(False)

    def register_module(self, name: str, module: Module) -> Module:
        """Register a module to the module.

        The modules will be registered in the order that this method is called.

        :param name: The name of the module.
        :param module: The module to register.

        :return: The module itself.

        raises:
            TypeError: If the module is not a Module subclass.
            KeyError: If the module with the same name already exists.
        """
        if not isinstance(module, Module):
            raise TypeError(f"Class '{type(module).__name__}' is not a Module subclass")
        if name in self._modules:
            raise KeyError(f"Module with name '{name}' already exists")
        self._modules[name] = module
        return module

    def register_parameter(self, name: str, parameter: Parameter) -> Parameter:
        """Register a parameter to the module.

        The parameters will be registered in the order that this method is called.

        :param name: The name of the parameter.
        :param parameter: The parameter to register.

        :return: The parameter itself.

        raises:
            TypeError: If the parameter is not a Parameter subclass.
            KeyError: If a parameter or buffer with the same name already exists.
        """
        if not isinstance(parameter, Parameter):
            raise TypeError(f"Class '{type(parameter).__name__}' is not a Parameter subclass")
        if name in self._parameters:
            raise KeyError(f"Parameter with name '{name}' already exists")
        if name in self._buffers:
            raise KeyError(f"Buffer with name '{name}' already exists")
        self._parameters[name] = parameter
        return parameter

    def register_buffer(self, name: str, buffer: Buffer) -> Buffer:
        """Register a buffer to the module.

        The buffers will be registered in the order that this method is called.

        :param name: The name of the buffer.
        :param buffer: The buffer to register.

        :return: The buffer itself.

        raises:
            TypeError: If the buffer is not a Buffer subclass.
            KeyError: If a buffer or parameter with the same name already exists.
        """
        if not isinstance(buffer, Buffer):
            raise TypeError(f"Class '{type(buffer).__name__}' is not a Buffer subclass")
        if name in self._buffers:
            raise KeyError(f"Buffer with name '{name}' already exists")
        if name in self._parameters:
            raise KeyError(f"Parameter with name '{name}' already exists")
        self._buffers[name] = buffer
        return buffer

    def modules(self) -> list[Module]:
        """Get the registered modules.

        The modules will be returned in the order that they were registered.

        :return: A list of modules.
        """
        return self._modules.values()

    def named_modules(self) -> list[str, Module]:
        """Get the registered modules and their names.

        The modules will be returned in the order that they were registered.

        :return: A tuple of (name, module) pairs.
        """
        return self._modules.items()

    def parameters(self, *, include_submodules: bool = True, as_array: bool = True) -> list[Parameter | wp.array]:
        """Get the registered parameters.

        The parameters will be returned in the order that they were registered.

        :param include_submodules: Whether to include the parameters of registered the sub-modules.
        :param as_array: Whether to return the parameters as Warp arrays or as
            :py:class:`~warp_nn.modules.parameter.Parameter` instances.

        :return: A list of parameters.
        """
        parameters = [parameter.data if as_array else parameter for parameter in self._parameters.values()]
        if include_submodules:
            for module in self._modules.values():
                parameters += [
                    parameter.data if isinstance(parameter, Parameter) and as_array else parameter
                    for parameter in module.parameters(as_array=as_array)
                ]
        return parameters

    def named_parameters(self) -> list[str, Parameter]:
        """Get the registered parameters and their names.

        The parameters will be returned in the order that they were registered.

        :return: A list of (name, parameter) pairs.
        """
        return self._parameters.items()

    def buffers(self, *, include_submodules: bool = True, as_array: bool = True) -> list[Buffer | wp.array]:
        """Get the registered buffers.

        The buffers will be returned in the order that they were registered.

        :param include_submodules: Whether to include the buffers of the registered sub-modules.
        :param as_array: Whether to return the buffers as Warp arrays or as
            :py:class:`~warp_nn.modules.buffer.Buffer` instances.

        :return: A list of buffers.
        """
        buffers = [buffer.data if as_array else buffer for buffer in self._buffers.values()]
        if include_submodules:
            for module in self._modules.values():
                buffers += module.buffers(as_array=as_array)
        return buffers

    def named_buffers(self) -> list[str, Buffer]:
        """Get the registered buffers and their names.

        The buffers will be returned in the order that they were registered.

        :return: A list of (name, buffer) pairs.
        """
        return self._buffers.items()

    def state_dict(self, *, destination: dict[str, wp.array] | None = None, prefix: str = "") -> dict[str, wp.array]:
        """Get the state dictionary, which is a reference to all the parameters and buffers of the modules and sub-modules.

        :param destination: The destination dictionary to store the state dictionary.
            This argument is used for internal recursion and should not be set by the user.
        :param prefix: The prefix to add to the names of the parameters and modules.
            This argument is used for internal recursion and should not be set by the user.

        :return: The state dictionary.
        """
        if destination is None:
            destination = OrderedDict()
        # store parameters
        for name, parameter in self._parameters.items():
            destination[f"{prefix}{name}"] = parameter.data
        # store buffers
        for name, buffer in self._buffers.items():
            destination[f"{prefix}{name}"] = buffer.data
        # iterate over modules
        for name, module in self._modules.items():
            module.state_dict(destination=destination, prefix=f"{prefix}{name}.")
        return destination

    def load_state_dict(self, state_dict: dict[str, wp.array]) -> None:
        """Load a state dictionary into the module.

        :param state_dict: The state dictionary to load into the module.

        :raises NotImplementedError: If the state dictionary contains an unsupported type.
        """

        def _load_from_state_dict(dst, src):
            if isinstance(src, dict):
                for k in src:
                    _load_from_state_dict(dst[k], src[k])
            elif isinstance(src, wp.array):
                wp.copy(dst, src.to(dst.device))
            elif isinstance(src, np.ndarray):
                wp.copy(dst, wp.array(src, dtype=dst.dtype, device=dst.device))
            else:
                raise NotImplementedError(f"Unsupported type: {type(src)}")

        _load_from_state_dict(self.state_dict(), state_dict)

    def to(self, device: wp.Device) -> Module:
        """Move the module to the specified device.

        :param device: The device to move the module to.
        :return: The module itself.
        """
        device = parse_device(device)
        # the arrays cached by the module (e.g. the outputs of the forward pass) live on the previous device
        if device != getattr(self, "_device", None) and hasattr(self, "_cache"):
            self._cache.clear()
        self._device = device
        if hasattr(self, "_modules"):
            for module in self._modules.values():
                module.to(self.device)
        if hasattr(self, "_parameters"):
            for parameter in self._parameters.values():
                parameter.to(self.device)
        if hasattr(self, "_buffers"):
            for buffer in self._buffers.values():
                buffer.to(self.device)
        return self
