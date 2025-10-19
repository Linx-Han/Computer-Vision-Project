# autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """图像编码器：提取特征"""
    def __init__(self, in_channels=3, embedding_size=128):
        super(ImageEncoder, self).__init__()
        
        # Encoder: 224x224 -> 56x56 -> 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 56 -> 28
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 14 -> 7
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Flatten and Dense
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 256, 256)
        self.fc2 = nn.Linear(256, embedding_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImageDecoder(nn.Module):
    """图像解码器：重构图像"""
    def __init__(self, embedding_size=128, out_channels=3):
        super(ImageDecoder, self).__init__()
        
        self.fc1 = nn.Linear(embedding_size, 256)
        self.fc2 = nn.Linear(256, 7 * 7 * 256)
        
        # Decoder: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 14 -> 28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 28 -> 56
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 56 -> 112
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # 112 -> 224
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出0-1
        )
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 256, 7, 7)
        x = self.decoder(x)
        return x


class ImageAutoencoder(nn.Module):
    """完整的Autoencoder"""
    def __init__(self, in_channels=3, embedding_size=128):
        super(ImageAutoencoder, self).__init__()
        self.encoder = ImageEncoder(in_channels, embedding_size)
        self.decoder = ImageDecoder(embedding_size, in_channels)
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding