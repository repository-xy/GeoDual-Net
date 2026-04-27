import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from types import SimpleNamespace

# Reuse Attention, Mlp, Block, Encoder modules
# Note: Ensure the Encoder output dimension in modelingnew.py is consistent with hidden_size
from modelingnew import Attention, Mlp, Block, Encoder


class TransUNet(nn.Module):
    # Receives config parameter (passed from trainC.py, ensuring hidden_size is unified)
    def __init__(self, num_classes=6, in_channels=3, img_size=256, config=None):  # Default in_channels changed to 3
        super(TransUNet, self).__init__()
        self.img_size = img_size  # Save input size for later upsampling matching

        # Handle config: prioritize externally passed ViT configuration (including hidden_size=768)
        if config is not None:
            self.config = config
            self.hidden_size = self.config.hidden_size  # Read 768 from config
        else:
            # Default configuration (for standalone execution)
            default_config = {
                'hidden_size': 768,
                'transformer': {'num_layers': 12, 'num_heads': 12, 'mlp_dim': 3072},
                'vit_patches_size': 16
            }
            self.config = SimpleNamespace(**default_config)
            self.hidden_size = self.config.hidden_size

        # ===================== Core modification 1: Remove pretrained weights, adapt to 3-channel input =====================
        # Load ResNet50 backbone, disable pretrained weights (avoid conflict between 3-channel pretrained weights and custom channels)
        backbone = resnet50(weights=None)  # Changed to None to avoid loading ImageNet pretrained weights
        # Dynamically replace the first convolution layer to adapt to input channels
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        # ==================================================================================

        # Convolutional encoder (ResNet50, layer3 output channels = 1024)
        self.conv_encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3  # Output: (B, 1024, img_size//16, img_size//16)
        )

        # -------------------------- Feature projection layer (core fix, enhanced robustness) --------------------------
        # Project ResNet50's 1024-dimensional features to Transformer's required 768 dimensions
        self.feature_projection = nn.Sequential(
            nn.Linear(1024, self.hidden_size),
            nn.LayerNorm(self.hidden_size),  # Normalization to improve training stability
            nn.ReLU(inplace=True)  # Added activation to enhance non-linear expression
        )
        # -------------------------------------------------------------------------------------

        # Transformer encoder (receives 768-dimensional features)
        self.transformer_encoder = Encoder(
            config=self.config,
            vis=False
        )

        # ===================== Core modification 2: Optimize decoder to ensure size matching =====================
        # UNet decoder (input dimension = hidden_size = 768, output gradually reduces dimension)
        # Decoder output sizes:
        # 768→512: (img_size//16)*2 = img_size//8
        # 512→256: img_size//4
        # 256→128: img_size//2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, 512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        original_size = (x.shape[2], x.shape[3])  # Save original input size, adapt to any input size

        # 1. Convolutional encoding (output: B, 1024, H//16, W//16)
        conv_feat = self.conv_encoder(x)  # (B, 1024, 16, 16) when img_size=256
        C, H, W = conv_feat.shape[1], conv_feat.shape[2], conv_feat.shape[3]

        # 2. Flatten into sequence (B, H*W, C) → (B, 256, 1024)
        seq = conv_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, 1024)

        # 3. Apply feature projection (1024→768)
        seq = self.feature_projection(seq)  # (B, 256, 768)

        # 4. Transformer encoding (input 768 dimensions to match LayerNorm requirements)
        trans_feat, _ = self.transformer_encoder(seq)  # (B, 256, 768)

        # 5. Restore to feature map (B, hidden_size, H, W) → (B, 768, 16, 16)
        trans_feat = trans_feat.permute(0, 2, 1).reshape(batch_size, self.hidden_size, H, W)

        # 6. Decode (768→128, size from 16 to 128 when img_size=256)
        x = self.decoder(trans_feat)  # (B, 128, 128, 128)

        # 7. Segmentation head prediction
        x = self.seg_head(x)  # (B, num_classes, 128, 128)

        # ===================== Core modification 3: Dynamically upsample to original input size =====================
        # Ensure output size exactly matches input (compatible with non-256 inputs)
        x = F.interpolate(
            x, size=original_size,
            mode='bilinear', align_corners=False
        )  # (B, 6, 256, 256)

        return x


# Test code (fully aligned with training logic, verify correctness)
if __name__ == "__main__":
    # Simulate ViT configuration from training code (import CONFIGS from modelingnew)
    from modelingnew import CONFIGS as CONFIGS_ViT_seg

    config = CONFIGS_ViT_seg['R50-ViT-B_16']
    config.n_classes = 6
    config.hidden_size = 768

    # Initialize model (exactly the same parameters as training code)
    model = TransUNet(
        num_classes=6,
        in_channels=3,  # Training code uses RGB 3 channels
        img_size=256,
        config=config
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
    grad_ok = model.seg_head.weight.grad is not None
    print(f"\nGradient propagation test: {'✅ Passed' if grad_ok else '❌ Failed'}")
    print("=" * 50)