import torch
import torch.nn as nn

class MixerLayer(nn.Module):
    def __init__(self, S, C, D_c, D_s):
        super().__init__()
        self.gelu = nn.GELU()
        self.layer_norm1 = nn.LayerNorm(C)
        self.layer_norm2 = nn.LayerNorm(C)

        self.token_mixer = nn.Sequential(
            nn.Linear(S, D_s),
            self.gelu,
            nn.Linear(D_s, S)
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(C, D_c),
            self.gelu,
            nn.Linear(D_c, C)
        )

    def forward(self, x):
        U = torch.transpose(
            self.token_mixer(
                torch.transpose(self.layer_norm1(x), 1, 2)),
            1, 2
        ) + x
        Y = self.channel_mixer(self.layer_norm2(U)) + U
        return Y


class MLPMixer(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        layer_num,
        embedding_dimension,
        channel_mixer_dimension,
        token_mixer_dimension,
        H, W
    ):
        super().__init__()
        self.C = embedding_dimension
        self.D_c = channel_mixer_dimension
        self.D_s = token_mixer_dimension
        self.P = patch_size
        self.N = layer_num
        self.S = (H // self.P) * (W // self.P)

        self.E = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.C,
            kernel_size=self.P,
            stride=self.P
        )

        self.layers = nn.Sequential()
        for i in range(layer_num):
            self.layers.append(MixerLayer(self.S, self.C, self.D_c, self.D_s))

        self.layer_norm = nn.LayerNorm(self.C)
        self.feature_dim = self.C
    
    def embedding(self, x):
        x = self.E(x)
        x = x.flatten(2)
        return x.transpose(1, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.layer_norm(x)
        return x.mean(dim=1)


def mlpmixer_s_4(H=32, W=32):
    return MLPMixer(
        in_channels=3,
        patch_size=4,
        layer_num=8,
        embedding_dimension=256,
        channel_mixer_dimension=1024,
        token_mixer_dimension=128,
        H=H, W=W
    )


def mlpmixer_b_16(H=224, W=224):
    return MLPMixer(
        in_channels=3,
        patch_size=16,
        layer_num=12,
        embedding_dimension=768,
        channel_mixer_dimension=3072,
        token_mixer_dimension=384,
        H=H, W=W
    )

def mlpmixer_l_32(H=224, W=224):
    return MLPMixer(
        in_channels=3,
        patch_size=32,
        layer_num=24,
        embedding_dimension=1024,
        channel_mixer_dimension=4096,
        token_mixer_dimension=512,
        H=H, W=W
    )