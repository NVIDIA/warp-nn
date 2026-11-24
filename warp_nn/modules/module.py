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
from collections.abc import Callable, Hashable

import numpy as np
import warp as wp

from warp_nn.modules.buffer import Buffer
from warp_nn.modules.parameter import Parameter
from warp_nn.utils import KernelConfig, cache, get_kernel_config, parse_device


class Kernels:
    def __init__(self) -> None:
        """Kernels of a module by kernel configuration, created the first time a configuration is requested.

        Instances are created by the modules (see :py:attr:`Module.kernels`), and should not be instantiated directly.
        """
        self._factory: Callable[..., Any] | None = None
        self._entries: dict[KernelConfig, Any] = {}

    def __contains__(self, config: KernelConfig) -> bool:
        """Whether the kernels for a kernel configuration have been created.

        :param config: Kernel configuration.

        :return: Whether the kernels for the kernel configuration have been created.
        """
        return config in self._entries

    def __repr__(self) -> str:
        return f"Kernels(factory={self._factory!r}, configs={len(self._entries)})"

    def __len__(self) -> int:
        """Number of kernel configurations whose kernels have been created.

        :return: The number of kernel configurations whose kernels have been created.
        """
        return len(self._entries)

    @property
    def factory(self) -> Callable[..., Any] | None:
        """Kernel factory, or ``None`` if it has not been set.

        It is a function that creates the kernels for a kernel configuration, called as ``factory(config=config)``
        (bind its other arguments with :py:func:`functools.partial`). It should create them in unique Warp modules
        (``@wp.kernel(module="unique")``), so that the kernels of each configuration compile on their own.
        Setting it discards the kernels created by the previous factory (if any). Once set, it cannot be unset.

        :raises TypeError: If the factory set is not callable.
        """
        return self._factory

    @factory.setter
    def factory(self, factory: Callable[..., Any]) -> None:
        if not callable(factory):
            raise TypeError(f"The kernel factory must be callable, got {type(factory).__name__}")
        self._factory = factory
        self._entries.clear()

    def select(self, *, config: KernelConfig, key: Hashable | None = None) -> Any:
        """Select the kernel for a kernel configuration, creating (and keeping) the kernels on first request.

        Select the kernels with the configurations adapted to the launch shapes (see
        :py:meth:`~warp_nn.utils.config.KernelConfig.adapt`), so that the launch shapes with the same adapted
        configuration share them. If the factory creates a collection of kernels (e.g. kernel overloads indexed by
        number of dimensions and data type), give the key of the kernel to select it from the collection.

        :param config: Kernel configuration (usually adapted to the launch shape).
        :param key: Key of the kernel in the collection created by the factory (e.g. ``(ndim, dtype)``), which must
            then be subscriptable. If ``None``, what the factory created is returned.

        :return: The kernel (or what the factory created, if no key is given) for the kernel configuration.

        :raises RuntimeError: If the kernel factory has not been set.
        """
        if config not in self._entries:
            if self._factory is None:
                raise RuntimeError("The kernel factory has not been set")
            self._entries[config] = self._factory(config=config)
        kernels = self._entries[config]
        return kernels if key is None else kernels[key]


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
        if not hasattr(self, "_cache"):
            self._cache: cache.Cache = cache.acquire(self)  # idempotent: the same sub-cache for the same module
        if not hasattr(self, "_kernels_by_config"):
            self._kernels_by_config: Kernels = Kernels()
        if not hasattr(self, "_config"):
            self._config: KernelConfig = get_kernel_config()
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
    def config(self) -> KernelConfig:
        """Kernel configuration with which the module was created.

        It is read when the module is created (see :py:func:`~warp_nn.utils.kernel_config`).
        """
        return self._config

    @property
    def cache(self) -> cache.Cache:
        """Sub-cache of the module in the global cache (see :py:mod:`~warp_nn.utils.cache`).

        It holds the data cached by the module (e.g. the output arrays of the forward pass),
        which is cleared when the module is moved to another device.
        """
        return self._cache

    @property
    def kernels(self) -> Kernels:
        """Kernels of the module by kernel configuration (see :py:class:`~warp_nn.modules.module.Kernels`).

        The module sets its kernel factory (usually in the constructor), and selects its kernels for the
        kernel configurations adapted to the launch shapes (see :py:meth:`~warp_nn.utils.config.KernelConfig.adapt`).
        Unlike the data in :py:attr:`cache`, the kernels are kept when the cache is disabled or cleared,
        and when the module is moved to another device.

        .. note::

            Not every built-in module uses it: the modules whose kernels are not adapted to the launch shapes (e.g. the
            convolution, pooling and normalization layers) create their kernels on their own, and keep the kernel
            factory unset.

        .. code-block:: python

            # constructor
            self.kernels.factory = partial(create_kernels, include_bias=bias)
            # forward pass
            config = self.config.adapt(shape, self.device, policy=AdaptationPolicy.MATMUL)
            kernel = self.kernels.select(config=config)
        """
        return self._kernels_by_config

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
        :raises ValueError: If an array in the state dictionary has a different size than the array it is loaded into.
        """

        def _load_from_state_dict(dst, src, name=""):
            if isinstance(src, dict):
                for k in src:
                    _load_from_state_dict(dst[k], src[k], k)
                return
            # the copy only requires arrays of the same size (e.g. a bias with shape (n,) can be loaded into (n, 1)),
            # otherwise a smaller array would only overwrite the leading elements of the destination array
            if isinstance(src, (wp.array, np.ndarray)) and src.size != dst.size:
                raise ValueError(
                    f"Unable to load '{name}' with shape {tuple(src.shape)} into an array with shape {tuple(dst.shape)}"
                )
            if isinstance(src, wp.array):
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
