import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import os
import rasterio as rio
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# First make the architecture

def create_resnet34_encoder(in_channels=5, pretrained=True):
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
    return model

class UNetWithResNet34(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, pretrained=True):
        super(UNetWithResNet34, self).__init__()
        self.encoder = create_resnet34_encoder(in_channels, pretrained=pretrained)

        #self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, output_padding=1)  # Adjust this as needed

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
        #print(f"d4 shape: {d4.shape}, x4 shape: {x4.shape}")  # Debugging line
        d4 = torch.cat((d4, x4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        #print(f"d3 shape: {d3.shape}, x3 shape: {x3.shape}")  # Debugging line
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        #print(f"d2 shape: {d2.shape}, x2 shape: {x2.shape}")  # Debugging line
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        #print(f"d1 shape: {d1.shape}, x1 shape: {x1.shape}")  # Debugging line
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.decoder1(d1)

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
    
# Second set up functions for the data loader and dataset

# Augment the data function
def augment(image, mask):
    flip = np.random.choice([0, 1])
    rot = np.random.choice([0, 1, 2, 3])
    image = np.rot90(image, k=rot, axes=(1, 2))
    mask = np.rot90(mask, k=rot, axes=(1, 2))
    if flip:
        image = np.flip(image, axis=-1)
        mask = np.flip(mask, axis=-1)
    # Ensure the array is contiguous
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask

# Load pre-cropped images function
def load_entire_image(image_path, mask_path):
    try:
        with rio.open(image_path) as src:
            image = src.read()
        with rio.open(mask_path) as src:
            mask = src.read()
    except rio.RasterioIOError as e:
        # Handle file not found or read error
        print(f"Error loading image or mask: {e}")
        image, mask = None, None
    return image, mask

# Create a data loader for the network
class RasterDataset(Dataset):
    def __init__(self, txt_file, crop_size):
        self.crop_size = crop_size
        with open(txt_file, 'r') as file:
            self.mask_paths = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        # Replace 'gridded_masks' with 'gridded_images_normalized' to get the corresponding image path
        image_path = mask_path.replace('gridded_masks', 'gridded_images_normalized')

        # Load the images and masks
        image, mask = load_entire_image(image_path, mask_path)
        if image is None or mask is None:
            print(f"Failed to load image or mask at index {idx}")
            return None  # Handle this appropriately, depending on whether you want to skip this batch or handle it differently

        # Apply augmentation
        image, mask = augment(image, mask)
        image = image.astype('float32') / 255.0  # Normalize images to [0, 1]
        return {'image': image, 'mask': mask}

    
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
    def __init__(self, n_channels=5, n_classes=1, lr=1e-4):
        super().__init__()
        self.model = UNetWithResNet34(in_channels=n_channels, out_channels=n_classes)
        #self.loss_fn = torch.nn.CrossEntropyLoss()
        #self.loss_fn = torch.nn.BCEWithLogitsLoss()
        #self.loss_fn = MaskedBCEWithLogitsLoss()
        self.loss_fn = MaskedBCEWithLogitsLoss(no_data_value=255)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        masks = masks.float()  # Convert masks to float type
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
    


def get_args():
    parser = argparse.ArgumentParser(description="Train UNet Model on Satellite Images")
    parser.add_argument('--txt_dir', type=str, required=True, help='Directory containing the train.txt, val.txt, and test.txt')
    parser.add_argument('--log_dir', type=str, default='tb_logs', help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--crop_size', type=int, default=448, help='The size of the cropped images')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--ngpus', type=int, default='1', help='Number of gpus to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Setting up data loaders
    dataset_train = RasterDataset(txt_file=os.path.join(args.txt_dir, 'train.txt'), crop_size=args.crop_size)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    dataset_val = RasterDataset(txt_file=os.path.join(args.txt_dir, 'val.txt'), crop_size=args.crop_size)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    dataset_test = RasterDataset(txt_file=os.path.join(args.txt_dir, 'test.txt'), crop_size=args.crop_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize the model and trainer
    model = CanopyModel(n_channels=5, n_classes=1, lr=args.lr)
    # Setup TensorBoardLogger
    logger = TensorBoardLogger(args.log_dir, name='model_0802_full_lidar')

    callbacks=[]
    callbacks.append(pl.callbacks.ModelCheckpoint(args.log_dir,'weights.pt',monitor='val_loss'))
    callbacks.append(pl.callbacks.ModelCheckpoint(args.log_dir,'weights.last.pt',monitor=None))
    #gpus_list = [int(g) for g in args.gpus.split(',')]  # Convert GPU indices from string to list of integers
    # Set up your training strategy with find_unused_parameters set to True
    
    trainer = Trainer(
        logger=logger,
        devices= args.ngpus,  
        max_epochs=args.epochs,
        accelerator='gpu',
        strategy='ddp' if args.ngpus>1 else 'auto',
        callbacks = callbacks
    ) 


    # Start training and testing
    trainer.fit(model, train_loader, val_loader)
    print("Training Complete")
    trainer.test(model, test_loader)

