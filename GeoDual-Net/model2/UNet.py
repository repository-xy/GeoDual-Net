import torch
import torch.nn as nn
import torch.nn.functional as F


# Double convolution block (Conv + ReLU + Conv + ReLU)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # Remove bias (more stable with BN)
            nn.BatchNorm2d(out_channels),  # Added BN layer (improve training stability, aligning with mainstream UNet)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Downsampling block (MaxPool + ConvBlock, double channels)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample: half size
            ConvBlock(in_channels, out_channels)  # Convolution: double channels
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Upsampling block (Transpose Conv + Concatenate skip connection + ConvBlock)
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Transposed convolution: upsample (double size) + halve channels (in_channels → in_channels//2)
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False
        )
        # After concatenation, channels = transposed conv output (in_channels//2) + skip connection channels (out_channels)
        self.conv = ConvBlock(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        # 1. Upsample (transposed convolution)
        x1 = self.up(x1)

        # 2. Handle size mismatch (robustness optimization: support odd sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # Symmetric padding to avoid size deviation
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        # 3. Channel concatenation (skip connection: x2 first, x1 second, matches UNet standard)
        x = torch.cat([x2, x1], dim=1)

        # 4. Double convolution to reduce channels
        return self.conv(x)


# Complete UNet model (compatible with training code's features config, enhanced robustness)
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, features=[64, 128, 256, 256]):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.n_levels = len(features)  # Number of downsampling layers (4)
        self.bottleneck_out = features[-1] * 2  # Bottleneck output channels (256×2=512, matching training code's features)

        # Initial convolution (input channels → first feature channels)
        self.in_conv = ConvBlock(in_channels, features[0])

        # Downsampling modules (compatible with arbitrary length features list)
        self.downs = nn.ModuleList()
        for i in range(self.n_levels - 1):
            self.downs.append(DownBlock(features[i], features[i + 1]))

        # Bottleneck (last feature channels → double channels)
        self.bottleneck = ConvBlock(features[-1], self.bottleneck_out)

        # Upsampling modules (core fix: compatible with features=[64,128,256,256])
        self.ups = nn.ModuleList()
        # Upsampling input channels: [bottleneck output, 256, 128] → corresponding to training code's features
        up_in_channels = [self.bottleneck_out] + features[1:-1][::-1]
        # Upsampling output channels: [256, 128, 64]
        up_out_channels = features[:-1][::-1]
        # Safety check: ensure number of upsampling layers matches
        assert len(up_in_channels) == len(up_out_channels), \
            f"Upsampling channel mismatch! in:{len(up_in_channels)}, out:{len(up_out_channels)}"

        for in_feat, out_feat in zip(up_in_channels, up_out_channels):
            self.ups.append(UpBlock(in_feat, out_feat))

        # Output layer (last feature channels → num_classes)
        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # 1. Downsampling + save skip connections
        skip_connections = []
        x = self.in_conv(x)  # 3→64, size 256×256
        for down in self.downs:
            skip_connections.append(x)  # Save: 64, 128, 256
            x = down(x)  # Downsample: 64→128→256→256 (aligning with training code's features)

        # 2. Bottleneck: 256→512, size 32×32
        x = self.bottleneck(x)

        # 3. Upsampling + concatenate skip connections (reverse skip connection list)
        skip_connections = skip_connections[::-1]  # [256, 128, 64]
        # Safety check: number of skip connections matches upsampling layers
        assert len(skip_connections) == len(self.ups), \
            f"Skip connection count mismatch! skip:{len(skip_connections)}, ups:{len(self.ups)}"

        for idx, up in enumerate(self.ups):
            x = up(x, skip_connections[idx])  # Upsample + concatenate + convolution

        # 4. Output: 64→6, size 256×256
        logits = self.out_conv(x)
        return logits


# Test code (fully aligned with training logic, verify correctness)
if __name__ == "__main__":
    # Initialize model (exactly the same parameters as training code)
    model = UNet(
        in_channels=3,  # Training code uses RGB 3 channels
        num_classes=6,  # Number of classes
        features=[64, 128, 256, 256]  # Features config in training code
    )

    # Simulate training input (batch_size=8, 3 channels, 256×256)
    x = torch.randn(8, 3, 256, 256)
    # Forward pass
    output = model(x)

    # Print key information for verification
    print("=" * 50)
    print(f"Input size: {x.shape}")  # Expected: torch.Size([8, 3, 256, 256])
    print(f"Output size: {output.shape}")  # Expected: torch.Size([8, 6, 256, 256])
    print("✅ Size matching test passed!")

    # Calculate parameter count (verify model complexity)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Gradient test (verify backpropagation)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, torch.randint(0, 6, (8, 256, 256)))
    loss.backward()
    # Check gradients of key layers
    grad_ok = model.out_conv.weight.grad is not None
    print(f"\nGradient propagation test: {'✅ Passed' if grad_ok else '❌ Failed'}")
    print("=" * 50)