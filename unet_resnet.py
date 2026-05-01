import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision.models import resnet34, ResNet34_Weights
from pytorch_lightning.loggers import TensorBoardLogger
import lightning.pytorch as pl


def create_resnet34_encoder(in_channels=5, pretrained=True, crop_size=448):
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = resnet34(weights=weights)
    original_first_layer = model.conv1
    new_first_layer = nn.Conv2d(
        in_channels, 
        64, 
        kernel_size=7, 
        stride=2, 
        padding=3, 
        bias=False
    )
    # Transfer the weights from the pretrained model to the new first layer
    if pretrained:
        new_first_layer.weight.data[:, :3] = original_first_layer.weight.data.clone()
        new_first_layer.weight.data[:, 3:] = original_first_layer.weight.data[:, :2].mean(dim=1, keepdim=True).clone()
    model.conv1 = new_first_layer

    # Adjust the stride of the first maxpool layer to accommodate different input sizes
    if crop_size != 448:
        model.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

    return model

class UNetWithResNet34(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, pretrained=True, crop_size=448):
        super(UNetWithResNet34, self).__init__()
        self.encoder = create_resnet34_encoder(in_channels, pretrained=pretrained, crop_size=crop_size)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._block(128, 64)

        # Additional upsampling layer
        self.final_upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x2 = self.encoder.layer1(self.encoder.maxpool(x1))
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        d4 = self.upconv4(x5)
        d4 = torch.cat((d4, x4), dim=1)
        d4 = self.decoder4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.decoder1(d1)

        # Apply the final upsampling and convolution
        d1 = self.final_upconv(d1)
        return self.final_conv(d1)
    

    @staticmethod
    def _block(in_channels, features):
        """
        Defines a decoding block with Convolution, Batch Normalization, and ReLU activations.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )