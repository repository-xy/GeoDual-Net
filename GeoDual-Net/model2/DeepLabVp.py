import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList()
        # 1x1 convolution branch
        self.aspp_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # Atrous convolution branches (3 different rates)
        for rate in atrous_rates:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # ===================== Fixed channel count calculation error =====================
        # Concatenated channels = out_channels × (1x1 branch + number of atrous branches + global pooling branch)
        # i.e., out_channels × (1 + len(atrous_rates) + 1) = out_channels × (len(atrous_rates)+2)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5)  # Added Dropout to improve generalization
        )

    def forward(self, x):
        res = []
        # 1x1 + atrous convolution branches
        for block in self.aspp_blocks:
            res.append(block(x))
        # Global pooling branch (upsample to original size)
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        res.append(global_feat)
        # Concatenate all branches
        x = torch.cat(res, dim=1)
        # Channel compression
        x = self.project(x)
        return x


class DeeplabV3Plus(nn.Module):
    def __init__(self, num_classes=6, in_channels=3):  # Default changed to 3 channels (aligned with training code)
        super(DeeplabV3Plus, self).__init__()
        # ===================== Core modification 1: Remove pretrained weights, adapt to 3 channels =====================
        # Load ResNet50 backbone, disable pretrained weights (avoid conflict between 3-channel pretrained weights and custom channels)
        backbone = resnet50(weights=None)  # Changed to None to avoid loading ImageNet pretrained weights
        # Dynamically replace the first convolution layer to adapt to input channels
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize the new convolution layer weights (improve training convergence)
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        # ==============================================================================

        # Extract features from each stage
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ASPP module (process layer4 output of 2048 channels)
        self.aspp = ASPP(2048, 256, atrous_rates=[6, 12, 18])

        # Low-level feature fusion (layer1 output 256 channels → 48 channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder head (fuse 256+48=304 channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),  # Added Dropout to prevent overfitting
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Segmentation head (256 → num_classes)
        self.seg_head = nn.Conv2d(256, num_classes, kernel_size=1)

        # Initialize decoder and segmentation head weights (improve training convergence)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Save original input size (dynamically adapt to any input size, no longer fixed to 256×256)
        original_size = (x.shape[2], x.shape[3])

        # Backbone network extracts multi-scale features
        feat1 = self.layer1(x)  # (B, 256, H/4, W/4) - low-level features
        feat2 = self.layer2(feat1)  # (B, 512, H/8, W/8)
        feat3 = self.layer3(feat2)  # (B, 1024, H/16, W/16)
        feat4 = self.layer4(feat3)  # (B, 2048, H/32, W/32) - high-level features

        # ASPP processes high-level features (2048→256)
        aspp_feat = self.aspp(feat4)  # (B, 256, H/32, W/32)

        # Upsample ASPP features to low-level feature size (H/4, W/4)
        aspp_feat = F.interpolate(
            aspp_feat, size=feat1.shape[2:], mode='bilinear', align_corners=False
        )

        # Process low-level features (256→48)
        low_feat = self.low_level_conv(feat1)  # (B, 48, H/4, W/4)

        # Fuse high/low-level features (256+48=304)
        fused = torch.cat([aspp_feat, low_feat], dim=1)  # (B, 304, H/4, W/4)

        # Decoder fuses features (304→256)
        x = self.decoder(fused)  # (B, 256, H/4, W/4)

        # Segmentation head predicts (256→num_classes)
        x = self.seg_head(x)  # (B, num_classes, H/4, W/4)

        # ===================== Core modification 2: Dynamically upsample to original size =====================
        # No longer fixed to 256×256, adapt to any input size (e.g., 128/256/512)
        x = F.interpolate(
            x, size=original_size, mode='bilinear', align_corners=False
        )
        return x


# Full test code (verify dimensions, gradients, robustness)
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model (exactly the same parameters as training code)
    model = DeeplabV3Plus(
        num_classes=6,
        in_channels=3  # Training code uses RGB 3 channels
    ).to(device)

    # Test 1: Standard 256×256 input (commonly used in training code)
    test_input = torch.randn(8, 3, 256, 256).to(device)  # batch_size=8
    output = model(test_input)
    print("\n=== Standard size test ===")
    print(f"Input size: {test_input.shape}")
    print(f"Output size: {output.shape}")  # Expected: torch.Size([8, 6, 256, 256])
    assert output.shape == (8, 6, 256, 256), "Standard size output mismatch!"

    # Test 2: Non-standard size robustness (e.g., 512×512)
    test_input_large = torch.randn(2, 3, 512, 512).to(device)
    output_large = model(test_input_large)
    print("\n=== Non-standard size test ===")
    print(f"Input size: {test_input_large.shape}")
    print(f"Output size: {output_large.shape}")  # Expected: torch.Size([2, 6, 512, 512])
    assert output_large.shape == (2, 6, 512, 512), "Non-standard size output mismatch!"

    # Test 3: ASPP module dimension verification
    aspp_module = model.aspp
    aspp_input = torch.randn(1, 2048, 8, 8).to(device)  # feat4 size (256/32=8)
    aspp_output = aspp_module(aspp_input)
    print("\n=== ASPP module test ===")
    print(f"ASPP input size: {aspp_input.shape}")
    print(f"ASPP output size: {aspp_output.shape}")  # Expected: torch.Size([1, 256, 8, 8])
    assert aspp_output.shape == (1, 256, 8, 8), "ASPP module output mismatch!"

    # Test 4: Gradient propagation verification
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    target = torch.randint(0, 6, (8, 256, 256)).to(device)
    loss = loss_fn(output, target)
    loss.backward()

    # Check gradients of key layers
    grad_ok = all([
        model.seg_head.weight.grad is not None,
        model.aspp.project[0].weight.grad is not None,
        model.decoder[0].weight.grad is not None
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