# Model Checkpoints

Place the following checkpoint files in this directory:

## canopy_lidar_base.ckpt
The base UNet-ResNet34 model trained on LiDAR-derived canopy labels across multiple California cities.
This corresponds to `weights.pt.ckpt` from training.

## canopy_statewide_finetuned.ckpt
The final model used for statewide inference, finetuned on hand-annotated data from all California climate zones.
This corresponds to `statewide_finetune_2_checkpoints/epoch=146-val_loss=0.60.ckpt` from training.

