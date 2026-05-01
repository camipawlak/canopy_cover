import sys
import os
import numpy as np
import cv2
from canopy_model import CanopyModel
import torch
import torch.nn as nn
from torchvision import models, transforms
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision.models import resnet34
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import tqdm
from torch.nn.functional import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--text_dir', type=str, required=True, help='path to directory containing test.txt')
parser.add_argument('--log_dir', default='log', help='path to log directory')
parser.add_argument('--crop_size', type=int, default=448, help='The size of the cropped images')
parser.add_argument('--out_dir', default='preds', help='path to output directory')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
args = parser.parse_args()
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
        #image, mask = augment(image, mask)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        image = image.astype('float32') / 255.0  # Normalize images to [0, 1]
        return {'image': image, 'mask': mask}

test_ds = RasterDataset(txt_file=os.path.join(args.text_dir, 'test.txt'), crop_size=args.crop_size)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

ckpt_path = os.path.join(args.log_dir, 'weights.pt.ckpt'
                         )
model = CanopyModel.load_from_checkpoint(ckpt_path)

trainer = pl.Trainer(accelerator='gpu', devices=1)
print("Setup Complete")
test_logits = trainer.predict(model, dataloaders=test_dl)

os.makedirs(args.out_dir,exist_ok=True)

n = 0
pbar = tqdm.tqdm(total=len(test_ds),desc='Writing predictions')

import torch.nn.functional as F  # Import functional API for sigmoid

for logits_batch in test_logits:
    # Convert logits to probabilities using sigmoid
    probabilities = torch.sigmoid(logits_batch).detach().cpu().numpy()
    # Threshold probabilities to get binary mask outputs
    predictions = (probabilities > 0.65).astype(np.uint8)  # Assuming 0.5 as the threshold

    for idx, prediction in enumerate(predictions):
        #image_path = test_ds.image_paths[n]
        #mask_path = image_path.replace('gridded_images_normalized', 'gridded_masks')  # Use mask path to retain spatial info
        mask_path = test_ds.mask_paths[n]

        with rio.open(mask_path) as src:
            # Prepare metadata to write new raster file
            meta = src.meta
            meta.update(dtype=rio.uint8, count=1)  # Ensure data type and count reflect the new mask

            pred_path = os.path.join(args.out_dir, os.path.basename(mask_path))
            # Write the prediction as a new raster file with the same metadata as the mask
            with rio.open(pred_path, 'w', **meta) as dst:
                dst.write(prediction.squeeze(), 1)  # Write prediction data to the first band

        n += 1
        pbar.update(1)
pbar.close()
