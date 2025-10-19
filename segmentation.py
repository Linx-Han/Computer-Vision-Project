# segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """U-Net用于食物分割"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (下采样)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decoder (上采样)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)  # 256 = 128 (upconv) + 128 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # 输出层
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def conv_block(self, in_ch, out_ch):
        """卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)           # [B, 32, 224, 224]
        enc2 = self.enc2(self.pool(enc1))  # [B, 64, 112, 112]
        enc3 = self.enc3(self.pool(enc2))  # [B, 128, 56, 56]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))  # [B, 256, 28, 28]
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)  # [B, 128, 56, 56]
        dec3 = torch.cat([dec3, enc3], dim=1)  # [B, 256, 56, 56]
        dec3 = self.dec3(dec3)  # [B, 128, 56, 56]
        
        dec2 = self.upconv2(dec3)  # [B, 64, 112, 112]
        dec2 = torch.cat([dec2, enc2], dim=1)  # [B, 128, 112, 112]
        dec2 = self.dec2(dec2)  # [B, 64, 112, 112]
        
        dec1 = self.upconv1(dec2)  # [B, 32, 224, 224]
        dec1 = torch.cat([dec1, enc1], dim=1)  # [B, 64, 224, 224]
        dec1 = self.dec1(dec1)  # [B, 32, 224, 224]
        
        # 输出
        out = self.out(dec1)  # [B, 1, 224, 224]
        return torch.sigmoid(out)  # 输出0-1之间的mask


def generate_pseudo_masks(depth_images, threshold=0.5):
    """
    基于深度图生成伪标签mask
    用于训练U-Net
    """
    masks = []
    for depth in depth_images:
        # 归一化
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        # 阈值分割
        mask = (depth_norm < threshold).float()
        masks.append(mask)
    return torch.stack(masks)