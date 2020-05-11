# Written by Ufuk Bombar, with help of 
# the original tensorflow implementation:
# https://github.com/ImpactCrater/SRNet-D/blob/master/model.py
# 
# and pytorch implementation:
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py

import torch
import torch.nn as nn 

class Conv2dBlock(nn.Module):
    '''
        SAME padding if (stide=1, dilation=1)
    '''
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros',
                    batch_norm=False,
                    activation=None):
        super(Conv2dBlock, self).__init__()
        if activation is not None:
            assert activation in {'relu', 'tanh', 'prelu'}

        conv_list = [nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=kernel_size // 2, # SAME padding if (stide=1, dilation=1)
                                dilation=1, 
                                groups=groups, 
                                bias=bias, 
                                padding_mode=padding_mode)]
        if batch_norm:
            conv_list.append(nn.BatchNorm2d(num_features=out_channels))
        
        if activation == 'relu':
            conv_list.append(nn.ReLU())
        elif activation == 'tanh':
            conv_list.append(nn.Tanh())
        elif activation == 'prelu':
            conv_list.append(nn.PReLU())

        self.conv_block = nn.Sequential(*conv_list)


    def forward(self, X):
        return self.conv_block(X)


class SubPixelConv2d(nn.Module):
    '''
        SubPixel convolution!
    '''
    def __init__(self, 
                    in_channels, 
                    kernel_size, 
                    bias=True, 
                    padding_mode='zeros',
                    scaling_factor=2):
        super(SubPixelConv2d, self).__init__()

        conv_list = [nn.Conv2d(in_channels=in_channels, 
                                out_channels=in_channels * (scaling_factor ** 2), 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=kernel_size // 2, # SAME padding if all default (stide=1, dilation=1)
                                dilation=1, 
                                groups=1, 
                                bias=bias, 
                                padding_mode=padding_mode)]
        conv_list.append(nn.PixelShuffle(scaling_factor))
        conv_list.append(nn.PReLU())
        self.conv_block = nn.Sequential(*conv_list)

    def forward(self, X):
        return self.conv_block(X)

class ResidualBlock(nn.Module):
    '''
        Residual block
    '''
    def __init__(self, in_channels, kernel_size):
        super(ResidualBlock, self).__init__()

        self.block1 = Conv2dBlock(in_channels=in_channels, 
                                    out_channels=in_channels, 
                                    kernel_size=kernel_size,
                                    batch_norm=True, 
                                    activation='prelu')

        self.block2 = Conv2dBlock(in_channels=in_channels, 
                                    out_channels=in_channels, 
                                    kernel_size=kernel_size,
                                    batch_norm=True, 
                                    activation=None)

    def forward(self, X):
        P1 = X  # (N, C, w, h)
        P2 = self.block1(X)  # (N, C, w, h)
        P2 = self.block2(P2)  # (N, C, w, h)
        return P1 + P2

class SRResNet(nn.Module):
    def __init__(self, 
                    scaling_factor=2,
                    in_channels=3, 
                    depth_channels=64,
                    lkernel_size=9, 
                    skernel_size=3, 
                    block_count=14):
        super(SRResNet, self).__init__()
        assert scaling_factor in {2, 4, 8}

        self.block1 = Conv2dBlock(in_channels=in_channels, 
                                    out_channels=depth_channels, 
                                    kernel_size=lkernel_size,
                                    batch_norm=False, 
                                    activation='prelu')

        self.block2 = nn.Sequential(*[
            ResidualBlock(depth_channels, skernel_size) for i in range(block_count)])

        self.block3 = Conv2dBlock(in_channels=depth_channels, 
                            out_channels=depth_channels, 
                            kernel_size=skernel_size,
                            batch_norm=True, 
                            activation=None)

        n_subpixel_blocks = {2:1, 4:2, 8:3}[scaling_factor]

        self.block4 = nn.Sequential(*[
            SubPixelConv2d(depth_channels, skernel_size, True, 'zeros', 2) for i in range(n_subpixel_blocks)])

        self.block5 = Conv2dBlock(in_channels=depth_channels, 
                            out_channels=in_channels, 
                            kernel_size=lkernel_size,
                            batch_norm=False, 
                            activation='tanh')

    def forward(self, X):
        X = self.block1(X)

        P1 = X
        P2 = self.block2(X)
        P2 = self.block3(P2)
        P3 = P1 + P2
        P3 = self.block4(P3)
        P3 = self.block5(P3)

        return (P3 + 1) / 2

