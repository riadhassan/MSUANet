from collections import OrderedDict
import torch
import torch.nn as nn
from network_lib.BAM import BAM
import math
import numpy as np
from torch.distributions.uniform import Uniform
import copy

def multiResBlock(in_channels, features, name,kernel=1,stride=1):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=kernel,
                        padding=1,
                        bias=False
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=kernel,
                        padding=1,
                        bias=False,
                        stride=stride
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
            ]
        )
    )

def decblock(in_channels, features, name,stride=1,pool_k=2,pool_stride=2):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        stride=stride
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "TConv", nn.ConvTranspose2d(features, features, kernel_size=pool_k, stride=pool_stride))
            ]
        )
    )

def enblock(in_channels, features, name,stride=1,pool_k=2,pool_stride=2):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        stride=stride
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name+ "pool",nn.MaxPool2d(kernel_size=pool_k, stride=pool_stride))
            ]
        )
    )

class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x

def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


class FeatureDrop(nn.Module):
    def __init__(self):
        super(FeatureDrop, self).__init__()
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        # x = self.upsample(x)
        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        # x = self.upsample(x)
        return x


class U_AttentionDense(nn.Module):
    def __init__(self, dims, Features):
        super(U_AttentionDense, self).__init__()
        self.features = Features
        self.shape = []
        for c, w in zip(Features, dims):
            self.shape.append((1, c, w, w))
        self.downsamples = OrderedDict()
        for i, x in enumerate(dims):
            self.downsamples['down' + str(i)] = nn.Upsample(size=(x, x), mode='bilinear')

    def create_weight_list(self, weights):
        self.weight_list = []
        for i, feat in enumerate(self.features):
            weights_new = self.downsamples['down' + str(i)](weights)
            weight_stake = []
            for j in range(feat):
                weight_stake.append(weights_new)
            weights_new = torch.stack(weight_stake, dim=1)
            weights_new = torch.reshape(weights_new, self.shape[i])
            weights_new = weights_new.to("cuda")
            self.weight_list.append(weights_new)

    def forward(self, weights, skips):
        self.create_weight_list(weights)
        self.skip_list = []

        for sk, we in zip(skips, self.weight_list):
            self.skip_list.append(sk * we)
        return self.skip_list


class U_Attention(nn.Module):
    def __init__(self, bottleneck_dim, reduction_ratio=16, dilation_num=2, dilation_val=4):
        super(U_Attention, self).__init__()
        self.attention = nn.Sequential()
        self.features = bottleneck_dim
        self.shape = (1, bottleneck_dim, 8, 8)

        self.downsample = nn.Upsample(size=(8, 8), mode='bilinear')
        self.attention.add_module("attSoftmax", nn.Softmax2d())
        # self.attention.add_module("attDown",nn.Conv2d(self.features,self.features,kernel_size=2,stride=16))
        self.attention.add_module("Spatial_BAM", BAM(self.features))

    def forward(self, weights, bottleneck):
        weights = self.downsample(weights)
        weight_list = []
        for i in range(self.features):
            weight_list.append(weights)
        weights = torch.stack(weight_list, dim=1)
        weights = torch.reshape(weights, self.shape)

        weights = weights.to("cuda")
        # attention = self.attention(weights)
        new_bottleneck = weights * bottleneck
        return new_bottleneck