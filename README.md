# California Canopy Cover Maps
This repository holds the code to use NAIP imagery to create canopy cover maps in California.

## Overview
Training was done in two steps. The first model is trained off of LiDAR-generated canopy models. Then, we fine-tuned that model using hand-annotated data from each of California's climate zones.

Pre-trained model checkpoints and training chips are available on Dryad [ https://doi.org/10.5061/dryad.rjdfn2zs9].

Climate zone data comes from: McPherson, E. G., Xiao, Q., Van Doorn, N. S., De Goede, J., Bjorkman, J., Hollander, A., Boynton, R. M., Quinn, J. F., & Thorne, J. H. (2017). The structure, function and value of urban forests in California communities. Urban Forestry & Urban Greening, 28, 43–53. https://doi.org/10.1016/j.ufug.2017.09.013


## Step 1: Generate LiDAR training data
To generate training data using LiDAR, please see the LiDAR-processing repository from Dr. Jon Ventura. The scripts can be used in this order:

1. `get_laz.py`
        python get_laz.py --city "CITYNAME" --out "OUTPATH"
           get_laz.py requires a 3dep.gpkg index file of 3DEP LiDAR collections. Build this from https://usgs.entwine.io/ and pass the path via --threedep_gpkg.
3. `make_chm.py`
        python make_chm.py "PATH/points.laz" --output_srs "EPSG:26910" --delaunay
4. `merge_chms.py`
        python merge_chms.py "/PATH/CITYNAME/points.laz"
5. `make_canopy.py`
        python make_canopy.py "NAIP DATA VRT PATH" "CHM TIF PATH" "CANOPY TIF PATH" --ndvithresh 0.05

## Step 2: Train the model

python 1_creating_chips.py --shapefile-path "PATH" --csv-path "PATH/city_names_folder_utm_year.csv" --input-vrt-folder "PATH" --input-canopy-folder "PATH TO LIDAR CHMS" --output-chips-folder "PATH/training_chips" --climate-zone-shapefile "data/climate_zone_CA/climate_zone_3.shp"

### Build NAIP VRTs before normalizing (requires AWS credentials, see data/README.md)
python make_aws_vrts.py --output_dir "PATH/naip_vrts" --shapefile_dir "PATH/naip_shapefiles"

python 2_normalize_chips_aws.py "PATH/training_chips" --vrt_dir "PATH/naip_vrts" --overwrite

python 3_make_splits.py --root-dir "PATH/training_chips" --out_dir "PATH/splits" --train-ratio 0.8 --test-ratio 0.1 --val-ratio 0.1

python 4_train_unet_resnet.py --txt_dir "PATH/splits" --log_dir "PATH/logs" --epochs 50 --batch_size 16 --crop_size 448 --lr 0.0001 --ngpus 1

python 5_test.py --text_dir "PATH/splits" --log_dir "PATH/logs" --crop_size 448 --out_dir "PATH/out_dir/" --batch_size 16

## Step 3: Fine-tune on hand-annotated data

python 6_make_hand_annotated_splits.py --root-dir "PATH/hand_ann_training_data_by_cz" --train-ratio 0.8 --test-ratio 0.1 --val-ratio 0.1

python 7_finetune_model.py --input_dir "PATH/hand_ann_training_data_by_cz/" --climate_zone "statewide_finetune_2" --log_dir "PATH/logs" --crop_size 256 --batch_size 16 --epochs 150 --lr 0.0001 --ngpus 1
# climate_zone can also be set to a specific zone, e.g. "southern_california_coast", "inland_empire", etc.

## Step 4: Run inference

python 8_inference.py --text_file "PATH/naip_tiles.txt" --output_raster "PATH/output.tif" --model_checkpoint "checkpoints/canopy_statewide_finetuned.ckpt" --tile_size 2048 --overlap 32 --year 2020

NAIP imagery is accessed via the USDA requester-p