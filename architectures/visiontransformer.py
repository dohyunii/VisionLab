import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, D, d_h):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn(D, d_h))
        self.W_K = nn.Parameter(torch.randn(D, d_h))
        self.W_V = nn.Parameter(torch.randn(D, d_h))
        self.scale = d_h**0.5
    
    def forward(self, x):
        Q = torch.matmul(x, self.W_Q)
        K = torch.matmul(x, self.W_K)
        V = torch.matmul(x, self.W_V)
        score = torch.softmax(torch.matmul(Q, torch.transpose(K, 1, 2)) / self.scale, dim=2)
        return torch.matmul(score, V)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.self_attentions = nn.ModuleList([
            SelfAttention(D, D // h)
            for i in range(h)
        ])
        self.linear = nn.Linear(D, D)
    
    def forward(self, x):
        multi_head_attention = []
        for self_attention in self.self_attentions:
            multi_head_attention.append(self_attention(x))
        multi_head_attention = torch.cat(multi_head_attention, dim=2)
        return self.linear(multi_head_attention)

class TransformerBlock(nn.Module):
    def __init__(self, D, h, hidden_layer):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.multi_head_attention = MultiHeadAttention(D, h)
        self.gelu = nn.GELU();

        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden_layer),
            self.gelu,
            nn.Linear(hidden_layer, D)
        )

    def forward(self, x):
        out = self.ln1(x)
        out = self.multi_head_attention(out)
        shortcut = x + out
        out = self.ln2(shortcut)
        out = self.mlp(out)
        return shortcut + out

class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        head_number,
        hidden_layer,
        n_layer,
        latent_vector_size,
        H, W
    ):
        super().__init__()
        self.D = latent_vector_size
        self.P = patch_size
        self.C = in_channels
        self.h = head_number
        self.N = (H // self.P) * (W // self.P)
        self.E = nn.Conv2d(
            in_channels=self.C,
            out_channels=self.D,
            kernel_size=self.P,
            stride=self.P
        )
        self.E_pos = nn.Parameter(torch.zeros(1, self.N + 1, self.D))

        self.x_class = nn.Parameter(torch.zeros(1, 1, self.D))

        self.encoder = nn.Sequential()
        for i in range(n_layer):
            self.encoder.append(TransformerBlock(self.D, self.h, hidden_layer))
        self.encoder.append(nn.LayerNorm(self.D))
        self.feature_dim = self.D
        

    def embedding(self, x):
        x = self.E(x)
        x = x.flatten(2)
        return x.transpose(1, 2)
    
    def forward(self, z):
        z = self.embedding(z)
        x_class = self.x_class.expand(z.shape[0], -1, -1)
        z = torch.cat((x_class, z), dim=1)
        z = z + self.E_pos
        z = self.encoder(z)
        return z[:, 0]


def vit_b_16(H=32, W=32):
    return VisionTransformer(
        in_channels=3,
        patch_size=16,
        head_number=12,
        hidden_layer=3072,
        n_layer=12,
        latent_vector_size=768,
        H=H,
        W=W
    )

def vit_l_32(H=32, W=32):
    return VisionTransformer(
        in_channels=3,
        patch_size=32,
        head_number=16,
        hidden_layer=4096,
        n_layer=24,
        latent_vector_size=1024,
        H=H,
        W=W
    )