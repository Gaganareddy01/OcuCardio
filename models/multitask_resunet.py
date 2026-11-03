import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    """
    Standard Double Convolution Block: [Conv2d -> BatchNorm -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Stage with Bilinear Upsampling and Skip Connection Fusion
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle padding mismatches if dimensions are slightly off
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SqueezeExcitation(nn.Module):
    """
    Channel Attention mechanism (SE-Block) to calibrate multi-scale retinal features.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


class RetinalMultiTaskResUNet(nn.Module):
    """
    Dual-Head Multi-Task ResNet-UNet Architecture for Retinal Image Analysis:
    --------------------------------------------------------------------------
    1. Shared Encoder: Pretrained ResNet-50 (or ResNet-34) extracting multi-scale
       representations of retinal geometry and microvascular architecture.
    2. Segmentation Head (U-Net Decoder):
       Outputs spatial masks for:
         - Channel 0: Optic Disc & Edema Boundary
         - Channel 1: Microaneurysms (MAs)
         - Channel 2: Neovascularization (NV)
         - Channel 3: Retinal Vasculature Bed (Arterioles / Venules)
    3. Classification / Risk Head (Systemic Vascular Risk):
       Fuses global bottleneck features with lesion-aware context to predict
       cardiovascular risk (e.g. High vs Low CVD risk or multi-class severity).
    """
    def __init__(
        self,
        num_seg_classes: int = 4,
        num_risk_classes: int = 1,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super(RetinalMultiTaskResUNet, self).__init__()

        self.num_seg_classes = num_seg_classes
        self.num_risk_classes = num_risk_classes

        # -----------------------------------------------------------
        # 1. ResNet Encoder Backbone
        # -----------------------------------------------------------
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            # Channel dimensions for ResNet-50
            c_in = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            # Channel dimensions for ResNet-34
            c_in = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'resnet34'.")

        # Initial stem
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # Output: C=64, H/2, W/2
        self.maxpool = resnet.maxpool # H/4, W/4

        # ResNet residual stages
        self.layer1 = resnet.layer1 # H/4, W/4
        self.layer2 = resnet.layer2 # H/8, W/8
        self.layer3 = resnet.layer3 # H/16, W/16
        self.layer4 = resnet.layer4 # H/32, W/32 (Bottleneck)

        # Attention block at bottleneck
        self.se_bottleneck = SqueezeExcitation(c_in[4])

        # -----------------------------------------------------------
        # 2. U-Net Segmentation Decoder
        # -----------------------------------------------------------
        self.dec4 = DecoderBlock(in_channels=c_in[4], skip_channels=c_in[3], out_channels=512) # H/16
        self.dec3 = DecoderBlock(in_channels=512, skip_channels=c_in[2], out_channels=256)      # H/8
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=c_in[1], out_channels=128)      # H/4
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=c_in[0], out_channels=64)       # H/2

        # Final upsampling to original input resolution (H, W)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(64, 32),
            nn.Conv2d(32, num_seg_classes, kernel_size=1)
        )

        # -----------------------------------------------------------
        # 3. CVD / Systemic Vascular Risk Classification Head
        # -----------------------------------------------------------
        # Combine global average and max pooling for richer microvascular feature statistics
        pooled_feat_dim = c_in[4] * 2

        self.risk_classifier = nn.Sequential(
            nn.Linear(pooled_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_risk_classes)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward Pass:
        x: [B, 3, H, W] - RGB Retinal Fundus Image

        Returns:
            seg_logits: [B, num_seg_classes, H, W] (raw logits for segmentation masks)
            risk_logits: [B, num_risk_classes] (raw logits for CVD risk score/class)
        """
        # Encoder forward pass
        s0 = self.stem(x)         # [B, 64, H/2, W/2]
        p0 = self.maxpool(s0)     # [B, 64, H/4, W/4]
        s1 = self.layer1(p0)      # [B, C1, H/4, W/4]
        s2 = self.layer2(s1)      # [B, C2, H/8, W/8]
        s3 = self.layer3(s2)      # [B, C3, H/16, W/16]
        s4 = self.layer4(s3)      # [B, C4, H/32, W/32]

        bottleneck = self.se_bottleneck(s4)

        # -----------------------------------------------------------
        # Segmentation Branch (U-Net)
        # -----------------------------------------------------------
        d4 = self.dec4(bottleneck, s3)  # [B, 512, H/16, W/16]
        d3 = self.dec3(d4, s2)          # [B, 256, H/8, W/8]
        d2 = self.dec2(d3, s1)          # [B, 128, H/4, W/4]
        d1 = self.dec1(d2, s0)          # [B, 64, H/2, W/2]
        seg_logits = self.final_upsample(d1)  # [B, num_seg_classes, H, W]

        # -----------------------------------------------------------
        # CVD Risk Classification Branch (ResNet Bottleneck Pooling)
        # -----------------------------------------------------------
        avg_pool = F.adaptive_avg_pool2d(bottleneck, (1, 1)).flatten(1)
        max_pool = F.adaptive_max_pool2d(bottleneck, (1, 1)).flatten(1)
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)  # [B, C4 * 2]

        risk_logits = self.risk_classifier(pooled_features)       # [B, num_risk_classes]

        return seg_logits, risk_logits
