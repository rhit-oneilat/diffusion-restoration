import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(32, out_ch) # Constraints: GroupNorm
        self.bnorm2 = nn.GroupNorm(32, out_ch)
        self.relu  = nn.SiLU() # Constraints: SiLU

    def forward(self, x, t, pool=False):
        # Time embedding
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

# Improved Block implementation for generic ResNet usage in U-Net
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1):
        super().__init__()
        self.project_time = None
        if time_emb_dim is not None:
             self.project_time = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), # Constraints: GroupNorm
            nn.SiLU(), # Constraints: SiLU
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t=None):
        h = self.block1(x)
        if self.project_time is not None and t is not None:
            # Time embedding injection
            # t shape: (batch, time_emb_dim) -> (batch, out_channels, 1, 1)
            time_emb = self.project_time(t)[:, :, None, None]
            h = h + time_emb
            
        h = self.block2(h)
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, head_size=64):
        super().__init__()
        self.channels = channels
        self.head_size = head_size
        self.heads = channels // head_size
        self.scale = head_size ** -0.5
        
        # If channels is not divisible by head_size, force 1 head or adjust
        if self.heads == 0:
            self.heads = 1
            self.head_size = channels
            self.scale = self.head_size ** -0.5
            
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1) # (B, H*W, C)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, -1, self.heads, self.head_size).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1, self.channels) # Back to (B, H*W, C)
        out = self.to_out(out)
        
        return out.permute(0, 2, 1).view(b, c, h, w) # Back to (B, C, H, W)
