import torch
import torch.nn as nn

class ConvMixerLayer(nn.Module):
    def __init__(self, h, k):
        super().__init__()
        self.gelu = nn.GELU()
        self.depthwise_conv = nn.Conv2d(
            in_channels=h,
            out_channels=h,
            kernel_size=k,
            stride=1,
            padding='same',
            groups=h
        )
        self.bn1 = nn.BatchNorm2d(h)
        self.pointwise_conv = nn.Conv2d(
            in_channels=h,
            out_channels=h,
            kernel_size=1
        )
        self.bn2 = nn.BatchNorm2d(h)

    def forward(self, x):
        out = x + self.bn1(self.gelu(self.depthwise_conv(x)))
        return self.bn2(self.gelu(self.pointwise_conv(out)))

class ConvMixer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        patch_size,
        embedding_dimension,
        kernel_size,
        layer_num
    ):
        super().__init__(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=embedding_dimension,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(embedding_dimension),
            *[ConvMixerLayer(embedding_dimension, kernel_size) for _ in range(layer_num)],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.feature_dim = embedding_dimension


def convmixer256_16():
    return ConvMixer(
        in_channel=3,
        patch_size=1,
        embedding_dimension=256,
        kernel_size=8,
        layer_num=16
    )

def convmixer768_32():
    return ConvMixer(
        in_channel=3,
        patch_size=14,
        embedding_dimension=768,
        kernel_size=7,
        layer_num=32
    )

def convmixer1536_20():
    return ConvMixer(
        in_channel=3,
        patch_size=7,
        embedding_dimension=1536,
        kernel_size=9,
        layer_num=20
    )