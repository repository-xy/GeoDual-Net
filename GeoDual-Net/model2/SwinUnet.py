import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


class Mlp(nn.Module):
    """MLP layer (inside Swin Transformer)"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition feature map into windows: (B, H, W, C) → (B*num_windows, window_size, window_size, C)"""
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"H/W must be divisible by window_size {window_size}, got H={H}, W={W}"
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Merge windows: (B*num_windows, window_size, window_size, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window attention layer (core of Swin)"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialization optimization
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Mask handling
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer basic block (fixed dynamic assignment of input_resolution)"""

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution if input_resolution is not None else (64, 64)  # default value to avoid None
        self.num_heads = num_heads
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.shift_size = shift_size if shift_size > 0 else 0
        self.mlp_ratio = mlp_ratio

        # Safety check: ensure window_size does not exceed input resolution
        H, W = self.input_resolution
        self.window_size = (min(self.window_size[0], H), min(self.window_size[1], W))
        self.shift_size = min(self.shift_size, self.window_size[0] // 2)

        # LayerNorm
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # DropPath layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Generate Attention Mask (adapted for dynamic resolution)
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size[0])
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature size {L} mismatch with resolution {H}*{W}"

        # Residual connection
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window attention computation
        x_windows = window_partition(shifted_x, self.window_size[0])
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size[0], H, W)

        # Inverse Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN residual
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch merging layer (downsampling, fixed dynamic assignment of input_resolution)"""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution if input_resolution is not None else (64, 64)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0, f"Input resolution {H}*{W} invalid"

        # Patch merging
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        # Normalization + dimension reduction
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """Swin Transformer Stage layer (fixed dynamic assignment of input_resolution)"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution if input_resolution is not None else (64, 64)
        self.depth = depth

        # Build Swin Blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=self.input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            blk.input_resolution = self.input_resolution  # Ensure block resolution matches layer resolution
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer (dynamic size adaptation)"""

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Dynamic patch embedding (compatible with any size)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


def DropPath(drop_prob=None):
    """DropPath layer (compatible with all PyTorch versions)"""

    class DropPathLayer(nn.Module):
        def __init__(self, drop_prob):
            super().__init__()
            self.drop_prob = drop_prob if drop_prob is not None else 0.0

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output

    return DropPathLayer(drop_prob) if drop_prob > 0. else nn.Identity()


class SwinTransformer(nn.Module):
    """Swin Transformer encoder (completely fixed input_resolution issue)"""

    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=0,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.window_size = window_size

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages (dynamically compute input_resolution)
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Dynamically compute resolution of current stage
            stage_resolution = (
                img_size // (patch_size * (2 ** i_layer)),
                img_size // (patch_size * (2 ** i_layer))
            )
            # Stage Blocks
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=stage_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer)
            self.layers.append(layer)

            # Downsample layer (none for last layer)
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(
                    input_resolution=stage_resolution,
                    dim=int(embed_dim * 2 ** i_layer),
                    norm_layer=norm_layer)
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(None)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch Embedding
        x = self.patch_embed(x)
        patch_H, patch_W = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]
        x = self.pos_drop(x)

        # Encoding process (dynamically update resolution)
        features = []
        current_H, current_W = patch_H, patch_W
        current_dim = self.embed_dim

        for i, (layer, downsample) in enumerate(zip(self.layers, self.downsamples)):
            # Update resolution of current stage
            layer.input_resolution = (current_H, current_W)
            # Execute blocks
            x = layer(x)
            # Save feature (convert to 4D tensor)
            feat = x.transpose(1, 2).contiguous().view(B, current_dim, current_H, current_W)
            features.append(feat)
            # Downsample
            if downsample is not None:
                downsample.input_resolution = (current_H, current_W)
                x = downsample(x)
                current_H, current_W = current_H // 2, current_W // 2
                current_dim *= 2

        return features  # [stage0, stage1, stage2, stage3]


class SwinUNet(nn.Module):
    """SwinUNet main class (adapted for trainV.py calls, no extra parameters)"""

    def __init__(self, num_classes=6, in_channels=3, img_size=256,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Encoder (fixed window_size=8)
        self.encoder = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=8,
            drop_path_rate=0.1,
            num_classes=0
        )

        # Decoder (dimensions strictly matched, added size adaptation)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(embed_dim * 4),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1, bias=True)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to improve training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoding: get 4 stage features
        features = self.encoder(x)

        # Decoding + skip connections (automatic size matching)
        # Stage3 → Stage2
        x = self.decoder1(features[3])
        x = self._resize_match(x, features[2])
        x = x + features[2]

        # Stage2 → Stage1
        x = self.decoder2(x)
        x = self._resize_match(x, features[1])
        x = x + features[1]

        # Stage1 → Stage0
        x = self.decoder3(x)
        x = self._resize_match(x, features[0])
        x = x + features[0]

        # Stage0 → original size
        x = self.decoder4(x)
        x = self._resize_match(x, (H, W))

        # Output segmentation result
        x = self.seg_head(x)
        return x

    @staticmethod
    def _resize_match(x, target):
        """Automatically adjust size to match target (compatible with tensor/tuple input)"""
        if isinstance(target, torch.Tensor):
            target_size = target.shape[2:]
        else:
            target_size = target
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


# Test code (verify all functionalities)
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model (exactly the same parameters as trainV.py)
    model = SwinUNet(
        num_classes=6,
        in_channels=3,
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    ).to(device)

    # Test 1: Standard 256×256 input
    test_input = torch.randn(8, 3, 256, 256).to(device)
    output = model(test_input)
    print("\n=== Standard size test ===")
    print(f"Input size: {test_input.shape}")
    print(f"Output size: {output.shape}")
    assert output.shape == (8, 6, 256, 256), "Standard size output mismatch!"

    # Test 2: Non-standard size robustness
    test_input_large = torch.randn(2, 3, 512, 512).to(device)
    output_large = model(test_input_large)
    print("\n=== Non-standard size test ===")
    print(f"Input size: {test_input_large.shape}")
    print(f"Output size: {output_large.shape}")
    assert output_large.shape == (2, 6, 512, 512), "Non-standard size output mismatch!"

    # Test 3: Encoder feature dimension verification
    features = model.encoder(test_input)
    print("\n=== Encoder feature dimensions ===")
    expected_dims = [96, 192, 384, 768]
    expected_sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    for i, (f, dim, size) in enumerate(zip(features, expected_dims, expected_sizes)):
        print(f"Stage{i}: dim={f.shape[1]}, size={f.shape[2:]}, expected={dim}/{size}")
        assert f.shape[1] == dim and f.shape[2:] == size, f"Stage{i} feature dimension mismatch!"

    # Test 4: Gradient propagation verification
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    target = torch.randint(0, 6, (8, 256, 256)).to(device)
    loss = loss_fn(output, target)
    loss.backward()

    # Check gradients of key layers
    grad_ok = all([
        model.seg_head.weight.grad is not None,
        model.decoder1[0].weight.grad is not None,
        model.encoder.layers[0].blocks[0].attn.proj.weight.grad is not None
    ])
    print("\n=== Gradient propagation test ===")
    print(f"Gradient propagation: {'✅ Passed' if grad_ok else '❌ Failed'}")
    assert grad_ok, "Gradient propagation failed!"

    # Test 5: Parameter count statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Parameter count statistics ===")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    print("\n✅ All tests passed!")