import torch
import torch.nn as nn
from networks.model_utils import *
from networks.sngan.snlayers.snconv2d import SNConv2d

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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            # nn.Linear(z_dim,64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),
            
            nn.Linear(64, 128),
            nn.ReLU(True),
            
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


class GANEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        architecture = {
            "conv_layers": 5,
            "conv_channels": [16, 32, 64, 128, 256],
            "conv_kernel_sizes": [(4, 4), (4, 4), (4, 4), (4,4), (4,4)],
            "conv_strides": [(1, 1), (2, 2), (1, 1), (2,2), (2,2)],
            "conv_paddings": [(1, 1), (1, 1), (1, 1), (1,1), (1,1)],
            "z_dimension": 64
        }
        input_shape = [1,51,51]
        self.main, self.output_shapes = create_encoder(architecture, input_shape)
        self.main.add_module('flatten',Flatten())
        encoded_shape = architecture['conv_channels'][-1]*np.prod(self.output_shapes[-1][:])
        self.main.add_module('lin_1', nn.Linear(encoded_shape, architecture['z_dimension']))
        def forward(self, input):
            return self.main(input)
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        ndf = 16
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 51 x 51
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
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
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class SNDiscriminator(nn.Module):
    def __init__(self):
        ndf = 16
        super(SNDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 51 x 51
            SNConv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 25 x 25
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 12
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 6
            SNConv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 3
            SNConv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)