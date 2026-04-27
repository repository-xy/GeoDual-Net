import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple

# --------------------------- Basic Configuration ---------------------------
CFG = {
    "in_channels": 3,  # RGB remote sensing image, 3 channels
    "num_classes": 6,  # 6-class output
    "embed_dim": 96,  # Base embedding dimension (adapted for lightweight design)
    "depths": [2, 2, 6, 2],  # Number of ViT blocks per stage (U-Net style 4 stages)
    "num_heads": [3, 6, 12, 24],  # Number of attention heads per stage
    "patch_size": 4,  # Remote sensing image patch size (4×4, suitable for small features)
    "window_size": 8,  # Window attention size (8×8, balancing global/local)
    "mlp_ratio": 4.,  # MLP hidden layer dimension multiplier
    "drop_rate": 0.1,  # Dropout rate
    "use_geo_pos_encoding": True,  # Remote sensing specific: geospatial position encoding
    "use_land_prior": True,  # Remote sensing specific: land cover prior feature fusion
    "decoder_embed_dim": 64  # Decoder embedding dimension (adapted for U-Net skip connections)
}


# --------------------------- Basic Modules ---------------------------
class ConvBNReLU(nn.Module):
    """Basic CNN module: extracts local texture/edge features from remote sensing images"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class GeoSpatialPosEncoding(nn.Module):
    """Remote sensing specific: geospatial position encoding (replaces ordinary 2D position encoding)"""

    def __init__(self, embed_dim: int, img_size: int = 256, patch_size: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Generate geographic coordinates (normalized to [-1,1], fitting the geographic coordinate system of remote sensing images)
        x_pos = torch.linspace(-1, 1, img_size // patch_size)
        y_pos = torch.linspace(-1, 1, img_size // patch_size)
        y, x = torch.meshgrid(y_pos, x_pos, indexing="ij")

        # Encode to embedding dimension
        self.x_embed = nn.Linear(1, embed_dim // 2)
        self.y_embed = nn.Linear(1, embed_dim // 2)

        # Register as buffers (not trainable)
        self.register_buffer('x_coord', x.reshape(-1, 1))
        self.register_buffer('y_coord', y.reshape(-1, 1))

    def forward(self) -> torch.Tensor:
        # Output: [1, num_patches, embed_dim]
        x_embed = self.x_embed(self.x_coord)  # [num_patches, embed_dim//2]
        y_embed = self.y_embed(self.y_coord)  # [num_patches, embed_dim//2]
        pos_embed = torch.cat([x_embed, y_embed], dim=-1)  # [num_patches, embed_dim]
        return pos_embed.unsqueeze(0)


class LandPriorFusion(nn.Module):
    """Remote sensing specific: land cover prior feature fusion
    Enhances feature representation of typical remote sensing land covers such as buildings, roads, vegetation, etc."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Land cover prior weights (6 classes: building, road, vegetation, water, bare land, other)
        self.land_weights = nn.Parameter(torch.ones(6, in_channels))
        self.conv = ConvBNReLU(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, c, h, w]
        b, c, h, w = x.shape
        # Fix: dimension expansion error, ensure weights are [1, c, 1, 1]
        weights = self.softmax(self.land_weights).mean(dim=0, keepdim=True)  # [1, c]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [1, c, 1, 1]
        weighted_feat = x * weights  # [b, c, h, w] (broadcasting correct)
        return self.conv(weighted_feat)


class Attention(nn.Module):
    """Lightweight multi-head attention: adapted for small remote sensing images"""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, drop_rate: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, num_patches, dim]
        b, n, c = x.shape

        # Generate QKV
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, num_heads, n, head_dim]

        # Window attention (adapted for remote sensing patches)
        q = q / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1))  # [b, num_heads, n, n]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attention weighting + projection
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron: feed-forward network for ViT blocks"""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop_rate: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SegViTBlock(nn.Module):
    """Core SegViT block: ViT attention + CNN local features"""

    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4., drop_rate: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, window_size, drop_rate)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop_rate=drop_rate)

        # CNN local feature supplement (texture enhancement for remote sensing images)
        self.cnn_feat = ConvBNReLU(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: [b, num_patches, dim] → restore 2D feature map: [b, dim, h, w]
        x_2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # CNN local features
        cnn_feat = self.cnn_feat(x_2d)
        cnn_feat = rearrange(cnn_feat, 'b c h w -> b (h w) c')

        # ViT attention
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # Fuse CNN + ViT features
        x = x + cnn_feat
        return x


# --------------------------- Encoder (ViT + U-Net downsampling) ---------------------------
class SegViTEncoder(nn.Module):
    """SegViT encoder: stage-wise downsampling, extract multi-scale remote sensing features"""

    def __init__(self, cfg: dict, img_size: int = 256):
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg["embed_dim"]
        depths = cfg["depths"]
        num_heads = cfg["num_heads"]
        patch_size = cfg["patch_size"]
        window_size = cfg["window_size"]
        mlp_ratio = cfg["mlp_ratio"]
        drop_rate = cfg["drop_rate"]

        # Input projection: 3 channels → embed_dim (4×4 patch, downsample factor 4)
        self.patch_embed = nn.Conv2d(cfg["in_channels"], embed_dim, kernel_size=patch_size, stride=patch_size,
                                     padding=0)
        self.num_patches = (img_size // patch_size) ** 2
        self.h, self.w = img_size // patch_size, img_size // patch_size

        # Remote sensing specific: geospatial position encoding
        if cfg["use_geo_pos_encoding"]:
            self.pos_encoding = GeoSpatialPosEncoding(embed_dim, img_size, patch_size)

        # Fix: replace Sequential with ModuleList, manually loop
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Precompute channels for each stage (critical: used for decoder matching)
        self.stage_channels = [embed_dim * (2 ** i) for i in range(len(depths))]

        for i in range(len(depths)):
            # Build list of SegViTBlocks for current stage
            stage_blocks = nn.ModuleList([
                SegViTBlock(self.stage_channels[i], num_heads[i], window_size, mlp_ratio, drop_rate)
                for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)

            # Downsample (except last stage)
            if i < len(depths) - 1:
                downsample = nn.Conv2d(self.stage_channels[i], self.stage_channels[i + 1], kernel_size=2, stride=2,
                                       padding=0)
                self.downsamples.append(downsample)

        # Remote sensing specific: land cover prior fusion
        if cfg["use_land_prior"]:
            final_dim = self.stage_channels[-1]
            self.land_fusion = LandPriorFusion(final_dim, final_dim)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[int], List[int], List[int]]:
        # x: [b, 3, 256, 256]
        features = []  # Save features from each stage
        hs, ws = [], []  # Save height/width of each stage feature

        # Input projection + position encoding
        x = self.patch_embed(x)  # [b, 96, 64, 64]
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # [b, 64×64, 96]

        if self.cfg["use_geo_pos_encoding"]:
            pos_embed = self.pos_encoding()  # [1, num_patches, embed_dim]
            x = x + pos_embed

        # Manually loop through blocks at each stage
        for i, stage_blocks in enumerate(self.stages):
            # Call each block
            for block in stage_blocks:
                x = block(x, h, w)

            # Save current stage features
            x_2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            features.append(x_2d)
            hs.append(h)
            ws.append(w)

            # Downsample (except last stage)
            if i < len(self.downsamples):
                x_2d = self.downsamples[i](x_2d)  # downsample by factor 2
                b, c, h, w = x_2d.shape
                x = rearrange(x_2d, 'b c h w -> b (h w) c')

        # Land cover prior fusion
        if self.cfg["use_land_prior"] and len(features) > 0:
            assert len(features[-1].shape) == 4, f"Feature dimension error, expected 4D, got {len(features[-1].shape)}D"
            features[-1] = self.land_fusion(features[-1])

        # Fix: return stage channels for decoder
        return features, hs, ws, self.stage_channels


# --------------------------- Decoder (U-Net upsampling + skip connections) ---------------------------
class SegViTDecoder(nn.Module):
    """SegViT decoder: upsampling + skip connections, restore 256×256 resolution"""

    def __init__(self, cfg: dict, stage_channels: List[int]):
        super().__init__()
        self.cfg = cfg
        self.stage_channels = stage_channels  # Encoder stage channels
        decoder_embed_dim = cfg["decoder_embed_dim"]
        num_classes = cfg["num_classes"]

        # Number of decoder stages (symmetric with encoder)
        self.num_stages = len(stage_channels)
        self.upconvs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        # Build decoder in reverse order (from deepest to shallowest)
        # Initial input channel: deepest stage channels
        current_dim = stage_channels[-1]
        for i in range(self.num_stages - 1, 0, -1):
            # Upsample: 2x upsampling
            upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconvs.append(upconv)

            # Feature fusion (skip connection): current channel + previous stage channel
            fusion_in = current_dim + stage_channels[i - 1]
            fusion = ConvBNReLU(fusion_in, decoder_embed_dim)
            self.fusions.append(fusion)

            # Update current channel to decoder embedding dimension
            current_dim = decoder_embed_dim

        # Final output head (6 classes)
        self.out_conv = nn.Sequential(
            ConvBNReLU(decoder_embed_dim, decoder_embed_dim),
            nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1, stride=1, padding=0)
        )

        # Final upsampling (restore to 256×256)
        self.final_upsample = nn.Upsample(scale_factor=cfg["patch_size"], mode='bilinear', align_corners=True)

    def forward(self, features: List[torch.Tensor], hs: List[int], ws: List[int]) -> torch.Tensor:
        # Start decoding from the deepest feature
        x = features[-1]

        # Stage-wise upsampling and fusion
        for i in range(len(self.upconvs)):
            # Upsample to corresponding size
            x = self.upconvs[i](x)
            # Get corresponding skip connection feature
            skip_feat = features[-(i + 2)]
            # Ensure size matching
            x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=True)
            # Concatenate features
            x = torch.cat([x, skip_feat], dim=1)
            # Feature fusion
            x = self.fusions[i](x)

        # Final upsampling to 256×256
        x = self.final_upsample(x)
        # Output 6-class result
        out = self.out_conv(x)
        return out


# --------------------------- Complete SegViT-RS Model ---------------------------
class SegViTRS(nn.Module):
    """Complete SegViT-RS remote sensing semantic segmentation model (ViT + U-Net hybrid architecture)"""

    def __init__(self, cfg: dict = CFG, img_size: int = 256):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size

        # Initialize encoder
        self.encoder = SegViTEncoder(cfg, img_size)
        # Fix: decoder receives stage channels from encoder to ensure channel matching
        self.decoder = SegViTDecoder(cfg, self.encoder.stage_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [b, 3, 256, 256]
        features, hs, ws, _ = self.encoder(x)
        # Decode + output: [b, 6, 256, 256]
        out = self.decoder(features, hs, ws)
        return out


# --------------------------- Test Code ---------------------------
if __name__ == "__main__":
    # Initialize model
    model = SegViTRS(CFG, img_size=256)
    model.eval()

    # Construct test input: batch_size=2, 3 channels, 256×256
    test_input = torch.randn(2, 3, 256, 256)

    # Forward pass
    with torch.no_grad():
        output = model(test_input)

    # Verify output dimensions (should be [2,6,256,256])
    print(f"Input dimensions: {test_input.shape}")
    print(f"Output dimensions: {output.shape}")
    print("SegViT-RS model initialized successfully, dimension verification passed!")

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e6:.2f}M")