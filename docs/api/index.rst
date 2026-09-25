API Reference
=============

.. toctree::
    :hidden:

    activations
    initializers
    layers
    operators
    optimizers
    utils

This section contains the API reference for the |warp-nn| library.

Core modules
------------

.. currentmodule:: warp_nn.modules
.. autosummary::
    :nosignatures:

    ~buffer.Buffer
    ~module.Module
    ~parameter.Parameter

Activations
-----------

.. currentmodule:: warp_nn.modules.activations
.. autosummary::
    :nosignatures:

    CELU
    ELU
    GELU
    HardSigmoid
    LeakyReLU
    LogSoftmax
    ReLU
    SELU
    Shrink
    Sigmoid
    Softmax
    SoftPlus
    SoftSign
    Swish
    Tanh
    Threshold

Initializers
------------

.. currentmodule:: warp_nn.initializers
.. autosummary::
    :nosignatures:

    constant
    kaiming_normal
    kaiming_uniform
    ones
    zeros

Layers
------

.. currentmodule:: warp_nn.modules.layers
.. autosummary::
    :nosignatures:

    AvgPool1D
    AvgPool2D
    BatchNorm
    Conv1D
    Conv2D
    Dropout
    Flatten
    GlobalAvgPool
    GlobalMaxPool
    GroupNorm
    GRUCell
    InstanceNorm
    LayerNorm
    Linear
    LSTMCell
    MaxPool1D
    MaxPool2D
    RMSNorm
    RNNCell
    Sequential

Operators
---------

.. currentmodule:: warp_nn.modules.operators
.. autosummary::
    :nosignatures:

    BinaryOp
    Clip
    UnaryOp

Optimizers
----------

.. currentmodule:: warp_nn.optimizers
.. autosummary::
    :nosignatures:

    Adam
    SGD
