import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms 
import torch.optim as optim
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter
from itertools import product
from slf_dataset import SLFDataset
import os

LR = 0.001
BATCH_SIZE = 20

# ROOT = '/home/pari/Projects/Research/Tensor_CS/data_drive/'
ROOT = '/nfs/stak/users/shressag/sagar/deep_completion/data/'

image_size = 51
nc = 2
ndf = 32
ngpu = 1

h_dim = 128
z_dim = 64


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))

    
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
                        # input is (nc) x 51 x 51
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten()
            # state size. (ndf*8) x 1 x 1
            # output state 
        )

    def forward(self, input):
#         for layer in self.main:
#             input = layer(input)
#             print(input.shape)
#         return input
        return self.main(input)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            nn.Linear(64, 128),
            UnFlatten((ndf*8, 1, 1)),
            # state size. (ndf*8) x 1 x 1
            nn.ConvTranspose2d(ndf*8, ndf*8, 3, 1, 0),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            
            # state size. (ndf*8) x 3 x 3
            nn.ConvTranspose2d(ndf*8, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            
            # state size. (ndf*4) x 6 x 6   
            nn.ConvTranspose2d(ndf*4, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),

            # state size (ndf*2) x 12 x 12
            nn.ConvTranspose2d(ndf*2, ndf*1, 4, 2, 0),
            nn.BatchNorm2d(ndf*1),
            nn.ReLU(True),

            # state size (ndf) x 26, 26
            nn.ConvTranspose2d(ndf*1,2, 4, 2, 0),
            nn.BatchNorm2d(2),
            nn.ReLU(True),

            # state size 2 x 54 x 54
            nn.Conv2d(2,1, 4, 1, 0),
            nn.Sigmoid()
            # output 1 x 51 x 51           
        )

    def forward(self, input):
#         for layer in self.main:
#             input = layer(input)
#             print(input.shape)
#         return input
        return self.main(input)


class VAE(nn.Module):
    def __init__(self, in_channels):
        super(VAE, self).__init__()
        self.h_dim = 128
        self.z_dim = 64
        if in_channels is None:
            self.enc = Encoder()
        else:
            self.enc = Encoder(in_channels)
        
        self.fc1 = nn.Linear(self.h_dim, self.z_dim)
        self.fc2 = nn.Linear(self.h_dim, self.z_dim)
        # self.fc3 = nn.Linear(self.z_dim, self.h_dim)
        
        self.dec = Decoder()
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def forward(self, x):
        h = self.enc(x)
        z, mu, logvar = self.bottleneck(h)
        # z = self.fc3(z)
        return self.dec(z), mu, logvar
    
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
