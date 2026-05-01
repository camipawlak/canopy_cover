import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from torchvision.models import resnet34, ResNet34_Weights
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from unet_resnet import UNetWithResNet34, create_resnet34_encoder

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, no_data_value=255):
        super().__init__()
        self.no_data_value = no_data_value

    def forward(self, outputs, targets):
        # Create a mask that will be True wherever targets are not equal to no_data_value
        mask = (targets != self.no_data_value)
        # Apply the mask
        outputs = outputs[mask]
        targets = targets[mask].float()  # Ensuring targets are float for BCE loss

        # Calculate the loss only on the masked data
        return nn.functional.binary_cross_entropy_with_logits(outputs, targets)

class CanopyModel(pl.LightningModule):
    def __init__(self, n_channels=5, n_classes=1, lr=1e-4, crop_size=448):
        super().__init__()
        self.model = UNetWithResNet34(in_channels=n_channels, out_channels=n_classes, crop_size=crop_size)
        #self.loss_fn = torch.nn.CrossEntropyLoss()
        #self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = MaskedBCEWithLogitsLoss(no_data_value=255)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        #masks = masks.long()  # Convert masks to Long type
        masks = masks.float()  # Convert masks to float type
        #masks = masks.unsqueeze(1)  # Add a channel dimension
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        #masks = masks.long()  # Convert masks to Long type
        masks = masks.float()  # Convert masks to float type
        #masks = masks.unsqueeze(1)  # Add a channel dimension
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log images, masks, and predictions to TensorBoard
        if batch_idx % 10 == 0:  # Log every 10 batches, adjust as needed
            # Ensure the images are in the correct format for TensorBoard
            # Assuming images and masks are already in [0, 1] range
            
            tb_images = images[:6]  # Select first 6 images to log
            images_rgb = tb_images[:, :3, :, :]
            tb_masks = masks[:6]
            tb_predictions = torch.sigmoid(outputs)[:6]  # Apply sigmoid to get probabilities for the first 6 predictions

            self.logger.experiment.add_images('val/images', images_rgb, self.current_epoch)
            self.logger.experiment.add_images('val/masks', tb_masks, self.current_epoch)
            self.logger.experiment.add_images('val/predictions', tb_predictions, self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        #masks = masks.long()  # Convert masks to Long type
        masks = masks.float()  # Convert masks to float type
        #masks = masks.unsqueeze(1)  # Add a channel dimension
        outputs = self(images)
        #loss = self.loss_fn(outputs, masks.squeeze(1))
        loss = self.loss_fn(outputs, masks)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['image']
        logits = self(images)  # These are raw logits
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        return probabilities
        