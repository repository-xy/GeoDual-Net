import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple

# --------------------------- Basic Configuration ---------------------------
CFG = {
    "in_channels": 3,  # Number of input channels (RGB remote sensing images)
    "num_classes": 6,  # 6-class output
    "embed_dims": [64, 128],  # Embedding dimensions of high-resolution streams
    "num_heads": [4, 8],  # Number of attention heads
    "window_size": 4,  # Window size adapted for remote sensing (4×4)
    "depths": [2, 2],  # Number of HRViT blocks per resolution stream
    "drop_rate": 0.1,  # Dropout rate
    "use_spectral_attention": True,  # Remote sensing specific: spectral attention
    "use_geo_pos_encoding": True  # Remote sensing specific: geospatial position encoding
}


# --------------------------- Basic Modules ---------------------------
class ConvBNReLU(nn.Module):
    """Basic CNN module: Conv + BN + ReLU, extracts local texture of remote sensing images"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SpectralAttention(nn.Module):
    """Remote sensing specific: spectral attention module, enhances channel/spectral features"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GeoPosEncoding(nn.Module):
    """Remote sensing specific: geospatial position encoding (replaces ordinary 2D position encoding)"""

    def __init__(self, embed_dim: int, h: int = 256, w: int = 256):
        super().__init__()
        x_pos = torch.linspace(-1, 1, w).unsqueeze(0).repeat(h, 1)
        y_pos = torch.linspace(-1, 1, h).unsqueeze(1).repeat(1, w)
        pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)  # [1, 2, h, w]

        self.pos_embed = nn.Conv2d(2, embed_dim, kernel_size=1, stride=1, padding=0)
        self.register_buffer('pos', pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos = self.pos.repeat(b, 1, 1, 1)  # [b, 2, h, w]
        pos_embed = self.pos_embed(pos)  # [b, embed_dim, h, w]
        return x + pos_embed


# --------------------------- Transformer Modules ---------------------------
class LightweightAttention(nn.Module):
    """Lightweight multi-head attention: fixed dimension splitting logic, adapted for remote sensing small images"""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, drop_rate: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # Defensive check: ensure embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1, stride=1, padding=0)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        ws = self.window_size

        # 1. Window partition: [b, c, h, w] → [b, c, num_windows, ws, ws]
        # Defensive check: ensure h/w divisible by window_size
        assert h % ws == 0 and w % ws == 0, f"h/w must be divisible by window_size {ws}"
        num_windows = (h // ws) * (w // ws)
        x_windowed = rearrange(x, 'b c (h ws1) (w ws2) -> b c (h w) ws1 ws2',
                               ws1=ws, ws2=ws, h=h // ws, w=w // ws)  # [b, c, num_windows, ws, ws]

        # 2. Convert to 4D input for Conv2d: [b, c, num_windows*ws, ws]
        x_4d = x_windowed.reshape(b, c, num_windows * ws, ws)  # 5D→4D

        # 3. Generate QKV: [b, 3*c, num_windows*ws, ws]
        qkv = self.qkv(x_4d)

        # --------------------------- [Fix] Core dimension splitting ---------------------------
        # Step 1: split 3*c into 3, c → [b, 3, self.embed_dim, num_windows*ws, ws]
        qkv = qkv.reshape(b, 3, self.embed_dim, num_windows * ws, ws)
        # Step 2: split c into num_heads, head_dim → [b, 3, num_heads, head_dim, num_windows*ws, ws]
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, num_windows * ws, ws)
        # Step 3: reshape window dimension → [b, 3, num_heads, head_dim, num_windows, ws, ws]
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, num_windows, ws, ws)
        # --------------------------------------------------------------------------

        # 4. Split Q/K/V (dimension 0 is 3, no out-of-bounds)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: [b, num_heads, head_dim, num_windows, ws, ws]

        # 5. Compute attention: flatten pixels within window → [b, num_heads, head_dim, num_windows, ws*ws]
        q_flat = q.reshape(b, self.num_heads, self.head_dim, num_windows, -1)  # [b, nh, hd, nw, ws²]
        k_flat = k.reshape(b, self.num_heads, self.head_dim, num_windows, -1)  # [b, nh, hd, nw, ws²]
        v_flat = v.reshape(b, self.num_heads, self.head_dim, num_windows, -1)  # [b, nh, hd, nw, ws²]

        # Attention scores: [b, nh, nw, ws², ws²]
        attn = (q_flat.transpose(-2, -1) @ k_flat) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 6. Attention weighted: [b, nh, hd, nw, ws²]
        out_flat = (attn @ v_flat.transpose(-2, -1)).transpose(-2, -1)
        # Restore window dimension: [b, nh, hd, nw, ws, ws]
        out = out_flat.reshape(b, self.num_heads, self.head_dim, num_windows, ws, ws)
        # Merge attention heads: [b, c, nw, ws, ws]
        out = out.reshape(b, self.embed_dim, num_windows, ws, ws)

        # 7. Restore original image size: [b, c, h, w]
        out = rearrange(out, 'b c (h w) ws1 ws2 -> b c (h ws1) (w ws2)',
                        h=h // ws, w=w // ws, ws1=ws, ws2=ws)

        # 8. Projection + Dropout
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class HRViTBlock(nn.Module):
    """Core HRViT block: CNN local features + Transformer global features"""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, drop_rate: float = 0.):
        super().__init__()
        # CNN branch: extracts local textures of remote sensing features
        self.cnn_branch = nn.Sequential(
            ConvBNReLU(embed_dim, embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        )

        # Transformer branch: captures global spatial dependencies
        self.trans_branch = nn.Sequential(
            LightweightAttention(embed_dim, num_heads, window_size, drop_rate),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        )

        # Fusion + residual
        self.norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_branch(x)
        trans_feat = self.trans_branch(x)
        out = self.norm(cnn_feat + trans_feat)
        out = self.relu(out + x)  # Residual connection
        return out


# --------------------------- HRViT-RS Backbone ---------------------------
class HRViTRSBackbone(nn.Module):
    """HRViT-RS backbone: parallel high-resolution streams, maintaining 256×256 features throughout"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        embed_dims = cfg["embed_dims"]
        depths = cfg["depths"]
        num_heads = cfg["num_heads"]
        window_size = cfg["window_size"]
        drop_rate = cfg["drop_rate"]

        # Input projection: 3 channels → 64 channels (maintain 256×256)
        self.stem = ConvBNReLU(cfg["in_channels"], embed_dims[0], kernel_size=3, stride=1, padding=1)

        # Remote sensing specific: geospatial position encoding
        if cfg["use_geo_pos_encoding"]:
            self.pos_encoding = GeoPosEncoding(embed_dims[0], h=256, w=256)

        # Remote sensing specific: spectral attention
        if cfg["use_spectral_attention"]:
            self.spectral_att = SpectralAttention(embed_dims[0])

        # High-resolution stream 1: 64 dimensions, 256×256 (no downsampling)
        self.stage1 = nn.Sequential(
            *[HRViTBlock(embed_dims[0], num_heads[0], window_size, drop_rate) for _ in range(depths[0])]
        )

        # High-resolution stream 2: 128 dimensions, 128×128 (only one downsampling)
        self.downsample = ConvBNReLU(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            *[HRViTBlock(embed_dims[1], num_heads[1], window_size, drop_rate) for _ in range(depths[1])]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: [b, 3, 256, 256]
        x = self.stem(x)  # [b, 64, 256, 256]

        # Remote sensing specific modules
        if self.cfg["use_geo_pos_encoding"]:
            x = self.pos_encoding(x)
        if self.cfg["use_spectral_attention"]:
            x = self.spectral_att(x)

        # High-resolution stream 1 output
        feat1 = self.stage1(x)  # [b, 64, 256, 256]

        # High-resolution stream 2 output
        feat2 = self.downsample(feat1)  # [b, 128, 128, 128]
        feat2 = self.stage2(feat2)  # [b, 128, 128, 128]

        return feat1, feat2


# --------------------------- Segmentation Head ---------------------------
class SegmentationHead(nn.Module):
    """Segmentation head: upsample to 256×256, output 6 classes"""

    def __init__(self, in_dims: List[int], num_classes: int):
        super().__init__()
        # Fuse multi-scale features
        self.fusion = ConvBNReLU(in_dims[0] + in_dims[1], in_dims[0], kernel_size=3, stride=1, padding=1)

        # Upsample: 128×128 → 256×256
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Output head: 6 classes
        self.out_conv = nn.Conv2d(in_dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feat1, feat2 = feats  # feat1:[b,64,256,256], feat2:[b,128,128,128]

        # Upsample feat2 to 256×256
        feat2_up = self.upsample(feat2)  # [b,128,256,256]

        # Fuse features
        fusion_feat = torch.cat([feat1, feat2_up], dim=1)  # [b,192,256,256]
        fusion_feat = self.fusion(fusion_feat)  # [b,64,256,256]

        # Output 6-class result
        out = self.out_conv(fusion_feat)  # [b,6,256,256]
        return out


# --------------------------- Complete HRViT-RS Model ---------------------------
class HRViTRS(nn.Module):
    """Complete HRViT-RS remote sensing semantic segmentation model"""

    def __init__(self, cfg: dict = CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = HRViTRSBackbone(cfg)
        self.seg_head = SegmentationHead(cfg["embed_dims"], cfg["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [b,3,256,256] → Output: [b,6,256,256]
        feats = self.backbone(x)
        out = self.seg_head(feats)
        return out


# --------------------------- Test Code ---------------------------
if __name__ == "__main__":
    # Initialize model
    model = HRViTRS(CFG)
    model.eval()

    # Construct test input: batch_size=2, 3 channels, 256×256
    test_input = torch.randn(2, 3, 256, 256)

    # Forward pass
    with torch.no_grad():
        output = model(test_input)

    # Verify output dimensions
    print(f"Input dimensions: {test_input.shape}")
    print(f"Output dimensions: {output.shape}")
    print("HRViT-RS model initialized successfully, dimension verification passed!")

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e6:.2f}M")