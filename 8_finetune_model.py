import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import rasterio as rio
from canopy_model import CanopyModel
import argparse

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing training data')
parser.add_argument('--climate_zone', type=str, required=True, help='Name of the climate zone')
parser.add_argument('--log_dir', default='log', help='Path to log directory')
parser.add_argument('--crop_size', type=int, default=256, help='The size of the cropped images')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs to use')
args = parser.parse_args()

# Function to load and augment images
def load_entire_image(image_path, mask_path):
    try:
        with rio.open(image_path) as src:
            image = src.read()
        with rio.open(mask_path) as src:
            mask = src.read()
    except rio.RasterioIOError as e:
        print(f"Error loading image or mask: {e}")
        return None, None
    return image, mask

def augment(image, mask):
    flip = np.random.choice([0, 1])
    rot = np.random.choice([0, 1, 2, 3])
    image = np.rot90(image, k=rot, axes=(1, 2))
    mask = np.rot90(mask, k=rot, axes=(1, 2))
    if flip:
        image = np.flip(image, axis=-1)
        mask = np.flip(mask, axis=-1)
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask

# Dataset class
class RasterDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mask_path = self.file_paths[idx]
        image_path = mask_path.replace('masks', 'normalized')
        image, mask = load_entire_image(image_path, mask_path)
        if image is None or mask is None:
            return None  # Consider handling this more gracefully
        image, mask = augment(image, mask)
        image = image.astype('float32') / 255.0
        #mask = mask.astype('float32') / 255.0
        return {'image': image, 'mask': mask}

def read_paths(base_dir, climate_zone, filename):
    # Construct the full path to the file
    full_path = os.path.join(base_dir, climate_zone, filename)
    with open(full_path, 'r') as file:
        # Read lines from the file, stripping any leading/trailing whitespace
        return [line.strip() for line in file.readlines()]

train_files = read_paths(args.input_dir, args.climate_zone, "train.txt")
val_files = read_paths(args.input_dir, args.climate_zone, "val.txt")
test_files = read_paths(args.input_dir, args.climate_zone, "test.txt")

# Setting up data loaders
train_dataset = RasterDataset(train_files)
val_dataset = RasterDataset(val_files)
test_dataset = RasterDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model, Trainer, and Logger setup
model = CanopyModel.load_from_checkpoint(os.path.join(args.log_dir, 'weights.pt.ckpt'))
logger = TensorBoardLogger(args.log_dir, name=f'model_{args.climate_zone}')
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.log_dir, f'{args.climate_zone}_checkpoints'),
                                      filename='{epoch}-{val_loss:.2f}',
                                      monitor='val_loss',
                                      mode='min',
                                      save_top_k=3)

trainer = Trainer(max_epochs=args.epochs,
                  logger=logger,
                  accelerator='gpu',
                  strategy='ddp' if args.ngpus > 1 else 'auto',
                  callbacks=[checkpoint_callback],
                  devices=args.ngpus)

# Start training and testing
trainer.fit(model, train_loader, val_loader)
print("Training Complete")
trainer.test(model, test_loader)
