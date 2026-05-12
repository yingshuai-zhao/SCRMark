import numpy as np
import torch.nn as nn
import torch
from .layers import *


class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super(ConvRelu, self).__init__()

        self.init_zero = init_zero
        if self.init_zero:
            self.layers = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)

        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                nn.LeakyReLU(inplace=False)
            )

    def forward(self, x):
        return self.layers(x)


class ConvTRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.layers(x)

class UpConvRelu(nn.Module):
    """
    二倍上采样
    """
    def __init__(self, channels_in, channels_out):
        super(UpConvRelu, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels_in, channels_out, 3, 1, padding=1, bias=True),
            nn.LeakyReLU(inplace=False)
        )

        self.up2 = ConvTRelu(channels_in, channels_out)

        self.conv = DoubleConvBNRelu(channels_out*2, channels_out)

    def forward(self, imgs):
        imgs_up1 = self.up1(imgs)
        imgs_up2 = self.up2(imgs)

        output = torch.cat([imgs_up1, imgs_up2], dim=1)

        return self.conv(output)

class ExpandNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ExpandNet, self).__init__()

        layers = [UpConvRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = UpConvRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class IndependentEncoder(nn.Module):
    def __init__(self, message_length=64, H=64, diffusion_length=256, channels=32):
        super(IndependentEncoder, self).__init__()

        self.diffusion_size = int(diffusion_length ** 0.5)

        self.linear1 = nn.Linear(message_length, diffusion_length)
        self.linear2 = nn.Linear(message_length, diffusion_length)
        self.linear3 = nn.Linear(message_length, diffusion_length)

        stride_blocks = int(np.log2(H // self.diffusion_size))

        self.message_pre_layer1 = nn.Sequential(
            ConvRelu(1, channels),
            ExpandNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )
        self.message_pre_layer2 = nn.Sequential(
            ConvRelu(1, channels),
            ExpandNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )
        self.message_pre_layer3 = nn.Sequential(
            ConvRelu(1, channels),
            ExpandNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )

    def forward(self, message):
        message_diff1 = self.linear1(message)
        message_diff2 = self.linear2(message)
        message_diff3 = self.linear3(message)

        message_image1 = message_diff1.view(-1, 1, self.diffusion_size, self.diffusion_size)
        message_image2 = message_diff2.view(-1, 1, self.diffusion_size, self.diffusion_size)
        message_image3 = message_diff3.view(-1, 1, self.diffusion_size, self.diffusion_size)
        
        message1 = self.message_pre_layer1(message_image1)
        message2 = self.message_pre_layer2(message_image2)
        message3 = self.message_pre_layer3(message_image3)
        
        return torch.concat([message1, message2, message3], dim=1)