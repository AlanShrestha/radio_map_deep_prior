"""
Implements the Original Variational Autoencoder paper: https://arxiv.org/pdf/1312.6114.pdf

Notation used:

    N: Batch Size
    C: Number of Channels
    H: Height (of picture)
    W: Width (of picture)
    z_dim: The dimension of the latent vector

"""
import sys

import multiprocessing

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from networks.model_utils import *
from networks.utils import *
import numpy as np

class Encoder(nn.Module):
    "Class that implements an encoder"

    def __init__(self, architecture, dataset_info):
        """
        param dict architecture:      A dictionary containing the hyperparameters that define the
                                         architecture of the model.
        :param dict hyperparameters:   A tuple that corresponds to the shape of the input.
        :param dict dataset_info:      The dimension of the latent vector z (bottleneck).

        The constructor of the Encoder
        """
        # call the constructor of the super class
        super(Encoder, self).__init__()

        # initialize class variables regarding the architecture of the model
        self.conv_layers = architecture["conv_layers"]
        self.conv_channels = architecture["conv_channels"]
        self.conv_kernel_sizes = architecture["conv_kernel_sizes"]
        self.conv_strides = architecture["conv_strides"]
        self.conv_paddings = architecture["conv_paddings"]
        self.z_dim = architecture["z_dimension"]

        # unpack the "dataset_info" dictionary
        self.dataset_method = dataset_info["ds_method"]
        self.dataset_shape = dataset_info["ds_shape"]
        self.dataset_path = dataset_info["ds_path"]

        # build the encoder
        self.encoder, self.encoder_shapes = create_encoder(architecture, self.dataset_shape)

        # compute the length of the output of the decoder once it has been flattened
        in_features = self.conv_channels[-1] * np.prod(self.encoder_shapes[-1][:])
        # now define the mean and standard deviation layers
        self.z_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)

    def forward(self, X):
        """
        :param Tensor X:  The input to run through the encoder. (N, C, H, W)

        :return: The output of the Encoder.
                   (N, z_dim)
        :rtype:  (Tensor)

        This method performs Forward Propagation through the encoder and outputs the latent vector
        """
        encoded_input = self.encoder(X)

        # flatten so that it can be fed to the mean and standard deviation layers
        encoded_input = torch.flatten(encoded_input, start_dim=1)

        # compute the mean and standard deviation
        z = self.z_layer(encoded_input)

        return z

class Decoder(nn.Module):
    "Class that implements an decoder"

    def __init__(self, architecture, dataset_info):
        """
        param dict architecture:      A dictionary containing the hyperparameters that define the
                                         architecture of the model.
        :param dict hyperparameters:   A tuple that corresponds to the shape of the input.
        :param dict dataset_info:      The dimension of the latent vector z (bottleneck).

        The constructor of the Encoder
        """
        # call the constructor of the super class
        super(Decoder, self).__init__()

        # initialize class variables regarding the architecture of the model
        self.conv_layers = architecture["conv_layers"]
        self.conv_channels = architecture["conv_channels"]
        self.conv_kernel_sizes = architecture["conv_kernel_sizes"]
        self.conv_strides = architecture["conv_strides"]
        self.conv_paddings = architecture["conv_paddings"]
        self.z_dim = architecture["z_dimension"]

        # unpack the "dataset_info" dictionary
        self.dataset_method = dataset_info["ds_method"]
        self.dataset_shape = dataset_info["ds_shape"]
        self.dataset_path = dataset_info["ds_path"]

        # build the encoder
        self.encoder, self.encoder_shapes = create_encoder(architecture, self.dataset_shape)

        # compute the length of the output of the decoder once it has been flattened
        in_features = self.conv_channels[-1] * np.prod(self.encoder_shapes[-1][:])

        # use a linear layer for the input of the decoder
        self.decoder_input = nn.Linear(in_features=self.z_dim, out_features=in_features)

        # build the decoder
        self.decoder = create_decoder(architecture, self.encoder_shapes)

        # build the output layer
        self.output_layer = create_output_layer(architecture, self.dataset_shape)

    def forward(self, z):
        """
        :param Tensor z:  The latent vector input to run through the decoder. (N, z_dim)

        :return: The output of the Encoder.
                   (N, C, H, W)
        :rtype:  (Tensor)

        This method performs Forward Propagation through the decoder and outputs the generated image
        """
        # run the latent vector through the "input decoder" layer
        decoder_input = self.decoder_input(z)

        # convert back the shape that will be fed to the decoder
        height = self.encoder_shapes[-1][0]
        width = self.encoder_shapes[-1][1]
        decoder_input = decoder_input.view(-1, self.conv_channels[-1], height, width)

        # run through the decoder
        decoder_output = self.decoder(decoder_input)

        # run through the output layer and return
        network_output = self.output_layer(decoder_output)
        return network_output

class Discriminator(nn.Module):
    """Class that implements the Decoder of Adversarial Autoencoder"""
    def __init__(self, z_dim):
        """
        :param dict architecture:      A dictionary containing the hyperparameters that define the
                                         architecture of the model.
        :param dict hyperparameters:   A tuple that corresponds to the shape of the input.
        :param dict dataset_info:      The dimension of the latent vector z (bottleneck).

        The constructor of the Variational Autoencoder.
        """

        # call the constructor of the super class
        super(Discriminator, self).__init__()
        self.z_dim = z_dim

        # build the discriminator
        self.discriminator = create_discriminator(self.z_dim)
        
    def forward(self, z):
        """
        :param Tensor z:  The latent vector input to run through the decoder. (N, z_dim)

        :return: The output of the Encoder.
                   (N, C, H, W)
        :rtype:  (Tensor)

        This method performs Forward Propagation through the discriminator and outputs the label(real, fake)
        """
        out = self.discriminator(z)
        return out

class AAE(nn.Module):
    """ Class that implements a Variational Autoencoder """

    def __init__(self, architecture, hyperparameters, dataset_info):
        """
        :param dict architecture:      A dictionary containing the hyperparameters that define the
                                         architecture of the model.
        :param dict hyperparameters:   A tuple that corresponds to the shape of the input.
        :param dict dataset_info:      The dimension of the latent vector z (bottleneck).

        The constructor of the Variational Autoencoder.
        """

        # call the constructor of the super class
        super(AAE, self).__init__()

        # initialize class variables regarding the architecture of the model
        self.conv_layers = architecture["conv_layers"]
        self.conv_channels = architecture["conv_channels"]
        self.conv_kernel_sizes = architecture["conv_kernel_sizes"]
        self.conv_strides = architecture["conv_strides"]
        self.conv_paddings = architecture["conv_paddings"]
        self.z_dim = architecture["z_dimension"]

        # unpack the "hyperparameters" dictionary
        self.batch_size = hyperparameters["batch_size"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.scheduler_step_size = hyperparameters["epochs"] // 2

        # unpack the "dataset_info" dictionary
        self.dataset_method = dataset_info["ds_method"]
        self.dataset_shape = dataset_info["ds_shape"]
        self.dataset_path = dataset_info["ds_path"]

        self.encoder = Encoder(architecture, hyperparameters, dataset_info)
        self.decoder = Decoder(architecture, hyperparameters, dataset_info)
        self.discriminator = Discriminator(architecture, hyperparameters, dataset_info)

        self.mse_loss = nn.MSELoss()
        
        b_size = t_slf.size(0)
        labels_real = torch.full((b_size,), real_label, device=run.device)
        labels_fake = torch.full((b_size,), fake_label, device=run.device)
            

    def train_step(self, X):
        # auto encoder step
        z_fake = self.encoder(X)
        X_hat = self.decoder(z_fake)
        ae_loss = self.mse_loss(X, X_hat)
        ae_loss.backward()

        # GAN step
        z_real = torch.randn_like(z_fake)
        
    