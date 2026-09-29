API Reference
=============

.. toctree::
    :hidden:

    activations
    initializers
    layers
    operators
    optimizers
    runtime
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
    HardSwish
    LeakyReLU
    LogSoftmax
    Mish
    PReLU
    ReLU
    SELU
    Shrink
    Sigmoid
    Softmax
    Softplus
    Softsign
    Swish
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
    Identity
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

    Abs
    Acos
    Acosh
    Add
    Asin
    Asinh
    Atan
    Atanh
    BitwiseAnd
    BitwiseNot
    BitwiseOr
    BitwiseXor
    Ceil
    Clip
    Cos
    Cosh
    Div
    Erf
    Exp
    Floor
    Log
    Max
    Min
    Mul
    Neg
    Pow
    Reciprocal
    Round
    Sign
    Sin
    Sinh
    Sqrt
    Sub
    Tan
    Tanh

Optimizers
----------

.. currentmodule:: warp_nn.optimizers
.. autosummary::
    :nosignatures:

    Adam
    SGD

Runtime
-------

.. currentmodule:: warp_nn.runtime
.. autosummary::
    :nosignatures:

    OnnxRuntimeV2
    OnnxRuntime
