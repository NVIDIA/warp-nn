Basics
======

This section walks through the main steps needed to define and train a neural network model with |warp-nn|:
defining a neural network, defining a loss function, running the forward pass and recording the kernel launches,
and calling the optimizer to update the model parameters.

.. seealso::

   Visit the :ref:`examples` section for working examples.

Defining a neural network
-------------------------

Import the necessary modules.

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-imports-start]
   :end-before: [basics-imports-end]

Subclass :py:class:`~warp_nn.modules.module.Module` and assign built-in layers as attributes.
Implement the :py:meth:`~warp_nn.modules.module.Module.__call__` method to define the forward pass of the model.

.. important::

    Call :py:meth:`~warp_nn.modules.module.Module.__post_init__` at the end of ``__init__`` so that sub-modules,
    parameters and buffers are registered automatically.

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-model-definition-start]
   :end-before: [basics-model-definition-end]

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-model-initialization-start]
   :end-before: [basics-model-initialization-end]

Defining a loss function
------------------------

Loss functions are plain Warp kernels that compute the value with respect to which differentiation will be carried out.

For example, to compute the Mean Squared Error (MSE) loss with ``"sum"`` reduction, accumulate the scalar loss into
a 1-element array with support for gradient accumulation (``requires_grad=True``).

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-loss-function-start]
   :end-before: [basics-loss-function-end]

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-loss-array-start]
   :end-before: [basics-loss-array-end]

Differentiation and optimization
--------------------------------

Instantiate the optimizer given the model's parameters.

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-optimizer-start]
   :end-before: [basics-optimizer-end]

Run the learning process iteratively across the dataset for the specified number of epochs.

.. important::

   Input arrays require the gradient to be enabled (``requires_grad=True``).

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-learning-loop-start]
   :end-before: [basics-learning-loop-end]

Wrap the forward pass and the loss kernel launch inside a :py:class:`~warp.Tape` context.
Every kernel launched inside the context block is recorded so that :py:meth:`~warp.Tape.backward`
can propagate gradients back through the model.

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-record-start]
   :end-before: [basics-record-end]

After each :py:meth:`~warp.Tape.backward` pass, call the optimizer's :py:meth:`~warp_nn.optimizers.Optimizer.step`
method to apply the gradient update. Call the tape's :py:meth:`~warp.Tape.zero` method afterwards to clear
the recorded operations and reset gradients before the next training iteration.

.. literalinclude:: ../snippets/basics.py
   :start-after: [basics-optimization-start]
   :end-before: [basics-optimization-end]
