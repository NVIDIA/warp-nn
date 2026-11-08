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

"""
This example demonstrates how to train and test a Variational AutoEncoder (VAE) on the MNIST dataset using Warp-NN.

The example illustrates the following concepts:
- How to define a custom module, as a :py:class:`~warp_nn.modules.module.Module` subclass, used for the reparameterization trick.

The implementation is organized through the following sections:
1. Custom module definition for the VAE's reparameterization trick.
2. VAE model definition.
3. Loss function definitions.
4. VAE model initialization and training/testing.
"""

import argparse
import time
import matplotlib.pyplot as plt
from mnist import MnistDataset
from utils import save_images

import numpy as np
import warp as wp

from warp_nn import nn, optimizers, utils


parser = argparse.ArgumentParser(description="Variational Autoencoder (VAE) - MNIST Example")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training/testing (default: 128)")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

# 1. --------------------------------------------------------------------


@wp.kernel
def _reparameterize_kernel(
    mean: wp.array2d(dtype=wp.float32),
    logvar: wp.array2d(dtype=wp.float32),
    seed: int,
    output: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    rng_state = wp.rand_init(seed, mean.shape[0] * i + j)
    output[i, j] = wp.randn(rng_state) * wp.exp(0.5 * logvar[i, j]) + mean[i, j]


class Reparameterize(nn.Module):
    def __init__(self, seed: int):
        """Reparameterization module for the VAE.

        This module moves the non-differentiable randomness operation outside the network by sampling random
        noise from a fixed distribution and transforming it. Doing so allows gradients to flow deterministically
        from the decoder to the encoder network.
        """
        super().__init__()
        self._seed = seed

    def __call__(self, mean, logvar):
        """Forward pass of the reparameterization module."""
        dtype = mean.dtype
        shape = mean.shape
        key = (shape, dtype)
        # cache output
        output = self.cache.get(key, lambda: wp.empty(shape, dtype=dtype, device=self.device, requires_grad=True))
        # launch kernel
        self._seed += 1
        wp.launch(_reparameterize_kernel, dim=shape, inputs=[mean, logvar, self._seed, output], device=self.device)
        return output


# 2. --------------------------------------------------------------------


class VAE(nn.Module):
    def __init__(self):
        """Variational AutoEncoder (VAE) model."""
        super().__init__()

        # define layers
        # - encoder layers
        self.fc1 = nn.Linear(784, 400)
        self.a1 = nn.ReLU()
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        # - decoder layers
        self.fc3 = nn.Linear(20, 400)
        self.a3 = nn.ReLU()
        self.fc4 = nn.Linear(400, 784)
        self.a4 = nn.Sigmoid()
        # - reparameterization layer
        self.reparameterize = Reparameterize(seed=args.seed)

        # register sub-modules and parameters
        super().__post_init__()

    def encode(self, x):
        """Map the input data ``x`` (i.e. MNIST image) to a latent space ``z`` described by a Gaussian distribution.

        The Gaussian probability distribution is parameterized by the mean ``mu`` and log-variance ``logvar``.
        Log-variance is used instead of standard deviation to bring stability and ease of training.
        """
        h1 = self.a1(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        """Map the latent space ``z`` to the output data (i.e. reconstructed MNIST image)."""
        h3 = self.a3(self.fc3(z))
        return self.a4(self.fc4(h3))

    def __call__(self, x):
        """Forward pass of the model during training/testing."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 3. --------------------------------------------------------------------


@wp.kernel
def _bce_loss(
    prediction: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    loss: wp.array1d(dtype=wp.float32),
):
    """Binary cross-entropy loss function."""
    i, j = wp.tid()
    wp.atomic_add(
        loss,
        0,
        -(
            target[i, j] * wp.max(wp.log(prediction[i, j]), -100.0)
            + (1.0 - target[i, j]) * wp.max(wp.log(1.0 - prediction[i, j]), -100.0)
        ),
    )


@wp.kernel
def _kld_loss(
    mu: wp.array2d(dtype=wp.float32),
    logvar: wp.array2d(dtype=wp.float32),
    loss: wp.array1d(dtype=wp.float32),
):
    """KL divergence loss function."""
    i, j = wp.tid()
    wp.atomic_add(loss, 0, -0.5 * (1.0 + logvar[i, j] - wp.pow(mu[i, j], 2.0) - wp.exp(logvar[i, j])))


# 4. --------------------------------------------------------------------

device = utils.parse_device(args.device)

# load the MNIST dataset for training and testing
train_dataset = MnistDataset(mode="train", batch_size=args.batch_size, device=args.device)
test_dataset = MnistDataset(mode="test", batch_size=args.batch_size, device=args.device)

# initialize the VAE model
model = VAE().to(device)

# initialize optimizer and loss
optimizer = optimizers.Adam(model.parameters(), lr=1e-3, device=device)
loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True, device=device)

train_loss = []
test_loss = []

for epoch in range(args.epochs):
    # training phase
    start_time = time.time()
    for i, (sample, _) in enumerate(train_dataset):
        # prepare sample (flatten images and enable gradients)
        sample = sample.reshape((sample.shape[0], -1))
        sample.requires_grad = True
        # compute loss and step optimizer
        loss.zero_()
        with wp.Tape() as tape:
            reconstruction, mu, logvar = model(sample)
            wp.launch(
                _bce_loss,
                dim=reconstruction.shape,
                inputs=[reconstruction, sample],
                outputs=[loss],
                device=device,
            )
            wp.launch(
                _kld_loss,
                dim=mu.shape,
                inputs=[mu, logvar],
                outputs=[loss],
                device=device,
            )
        tape.backward(loss)
        optimizer.step()
        tape.zero()
        # logging
        train_loss.append(loss.numpy().item() / sample.shape[0])
        msg = (
            f"Training | epoch {epoch + 1}/{args.epochs}, "
            f"batch: {i + 1}/{len(train_dataset)}, "
            f"loss: {train_loss[-1]:.2f}"
        )
        print(msg, end="\r")
    print(f"{msg} (time: {(time.time() - start_time):.2f} seconds)")

    # testing phase
    start_time = time.time()
    for i, (sample, _) in enumerate(test_dataset):
        # prepare sample (flatten images)
        sample = sample.reshape((sample.shape[0], -1))
        # compute loss
        loss.zero_()
        reconstruction, mu, logvar = model(sample)
        wp.launch(
            _bce_loss,
            dim=reconstruction.shape,
            inputs=[reconstruction, sample],
            outputs=[loss],
            device=device,
        )
        wp.launch(
            _kld_loss,
            dim=mu.shape,
            inputs=[mu, logvar],
            outputs=[loss],
            device=device,
        )
        # logging
        test_loss.append(loss.numpy().item() / sample.shape[0])
        msg = (
            f" Testing | epoch {epoch + 1}/{args.epochs}, "
            f"batch: {i + 1}/{len(test_dataset)}, "
            f"loss: {test_loss[-1]:.2f}"
        )
        print(msg, end="\r")
        # save reconstruction image
        if i == 0:
            n = min(sample.shape[0], 8)  # number of images to save
            comparison = np.concatenate(
                [sample.numpy()[:n].reshape((n, 28, 28)), reconstruction.numpy()[:n].reshape((n, 28, 28))], axis=0
            )
            save_images(comparison, f"results/reconstruction-epoch_{epoch + 1}.png", rows=2)
    print(f"{msg} (time: {(time.time() - start_time):.2f} seconds)")

    # generate sample images
    sample = np.random.randn(64, 20).astype(np.float32)
    result = model.decode(wp.array(sample, device=device))
    images = result.numpy().reshape(64, 28, 28)
    save_images(images, f"results/sample-epoch_{epoch + 1}.png")

# plot metrics
fig = plt.figure(figsize=(10, 5))
axes = fig.subplots(1, 2, sharey=True)
for axis, data, title in zip(axes, [train_loss, test_loss], ["Train", "Test"]):
    axis.set_title(title)
    axis.set_xlabel("Steps")
    axis.set_ylabel("Loss")
    axis.grid(True)
    axis.plot(data)
fig.suptitle("Train/test loss")
fig.savefig(f"results/loss.png")
plt.close(fig)

print("Done! Images and metrics saved in 'results' directory.")
