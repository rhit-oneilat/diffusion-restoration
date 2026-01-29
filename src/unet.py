import torch
import torch.nn as nn
from src.modules import SinusoidalPositionEmbeddings, ResNetBlock, SelfAttention

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        self.inc = nn.Conv2d(c_in, 64, kernel_size=3, padding=1)

        # Down 1
        self.down1_res1 = ResNetBlock(64, 128, time_emb_dim=time_dim)
        self.down1_res2 = ResNetBlock(128, 128, time_emb_dim=time_dim)
        self.down1_pool = nn.Conv2d(128, 128, 4, 2, 1) # 64 -> 32

        # Down 2
        self.down2_res1 = ResNetBlock(128, 256, time_emb_dim=time_dim)
        self.down2_res2 = ResNetBlock(256, 256, time_emb_dim=time_dim)
        self.down2_pool = nn.Conv2d(256, 256, 4, 2, 1) # 32 -> 16

        # Down 3
        self.down3_res1 = ResNetBlock(256, 256, time_emb_dim=time_dim)
        self.down3_res2 = ResNetBlock(256, 256, time_emb_dim=time_dim)
        self.down3_pool = nn.Conv2d(256, 256, 4, 2, 1) # 16 -> 8

        # Bottleneck
        self.bot1 = ResNetBlock(256, 512, time_emb_dim=time_dim)
        self.attn = SelfAttention(512)
        self.bot2 = ResNetBlock(512, 512, time_emb_dim=time_dim)
        self.bot3 = ResNetBlock(512, 256, time_emb_dim=time_dim)

        # Up 1
        self.up1_up = nn.ConvTranspose2d(256, 256, 4, 2, 1) # 8 -> 16
        # Input to res blocks will be cat(256, 256) = 512 if using skips.
        # Wait, the skip connection is with the output of the corresponding down block.
        # Down3 output (before pool) is 256. 
        # So we cat 256 (from up) + 256 (from down3 skip) = 512.
        self.up1_res1 = ResNetBlock(512, 256, time_emb_dim=time_dim)
        self.up1_res2 = ResNetBlock(256, 256, time_emb_dim=time_dim)

        # Up 2
        self.up2_up = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 16 -> 32
        # Skip from Down2 is 256.
        # Cat 128 (from up) + 256 (from down2 skip) = 384. 
        self.up2_res1 = ResNetBlock(384, 128, time_emb_dim=time_dim) 
        self.up2_res2 = ResNetBlock(128, 128, time_emb_dim=time_dim)

        # Up 3
        self.up3_up = nn.ConvTranspose2d(128, 64, 4, 2, 1) # 32 -> 64
        # Skip from Down1 is 128.
        # Cat 64 (from up) + 128 (from down1 skip) = 192.
        self.up3_res1 = ResNetBlock(192, 64, time_emb_dim=time_dim)
        self.up3_res2 = ResNetBlock(64, 64, time_emb_dim=time_dim)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x, t):
        t = t.type(torch.float)
        t = self.time_mlp(t)

        x = self.inc(x)
        
        # Encoder
        # Down 1
        x1 = self.down1_res1(x, t)
        x1 = self.down1_res2(x1, t) # Skip 1: 128ch, 64x64. Wait, shape is same as input? No convs preserve.
        x_skip1 = x1
        x = self.down1_pool(x1) # 32x32

        # Down 2
        x2 = self.down2_res1(x, t)
        x2 = self.down2_res2(x2, t) # Skip 2: 256ch, 32x32
        x_skip2 = x2
        x = self.down2_pool(x2) # 16x16

        # Down 3
        x3 = self.down3_res1(x, t)
        x3 = self.down3_res2(x3, t) # Skip 3: 256ch, 16x16
        x_skip3 = x3
        x = self.down3_pool(x3) # 8x8

        # Bottleneck
        x = self.bot1(x, t)
        x = self.attn(x)
        x = self.bot2(x, t)
        x = self.bot3(x, t) # 256ch, 8x8

        # Decoder
        # Up 1 (Target 16x16)
        x = self.up1_up(x) # 256ch, 16x16
        x = torch.cat([x, x_skip3], dim=1) # 256 + 256 = 512
        x = self.up1_res1(x, t)
        x = self.up1_res2(x, t)

        # Up 2 (Target 32x32)
        x = self.up2_up(x) # 128ch, 32x32
        x = torch.cat([x, x_skip2], dim=1) # 128 + 256 = 384
        x = self.up2_res1(x, t)
        x = self.up2_res2(x, t)

        # Up 3 (Target 64x64)
        x = self.up3_up(x) # 64ch, 64x64
        x = torch.cat([x, x_skip1], dim=1) # 64 + 128 = 192
        x = self.up3_res1(x, t)
        x = self.up3_res2(x, t)

        return self.outc(x)
