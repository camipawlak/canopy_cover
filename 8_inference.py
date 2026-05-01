
import os
import sys
import argparse
import numpy as np
import rasterio
import torch
import logging
import traceback
from canopy_model import CanopyModel
from skimage.transform import resize
from tqdm import tqdm, trange
from functools import lru_cache
import warnings
from skimage.exposure import match_histograms
from rasterio.warp import calculate_default_transform, reproject, Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("botocore").setLevel(logging.ERROR)
logging.getLogger("boto3").setLevel(logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.WARNING)

os.environ['AWS_REQUEST_PAYER'] = 'requester'

@lru_cache(maxsize=10)
def load_reference_image(reference_path):
    with rasterio.open(reference_path) as src_ref:
        raster_data_ref = src_ref.read().transpose(1, 2, 0)
    return raster_data_ref

def perform_histogram_matching(image, reference_path):
    try:
        raster_data_orig = image.transpose(1, 2, 0)
        raster_data_ref = load_reference_image(reference_path)

        # Perform histogram matching directly on the full-resolution images
        matched_image = match_histograms(
            raster_data_orig, raster_data_ref, channel_axis=2).astype(np.uint8)

        # Reshape back to the original format (bands, height, width)
        matched_image = matched_image.transpose(2, 0, 1)
        return matched_image
    except Exception as e:
        logging.error(f"Error in perform_histogram_matching: {e}")
        return image  # Return original image if matching fails

# Function to make chips from raster data for inference
def make_chips(data, tile_size, overlap):
    padded_size = tile_size + overlap * 2
    chips = []
    rows = []
    cols = []
    bands, height, width = data.shape
    if height < tile_size and width < tile_size:
        logging.warning("Data is smaller than the tile size. Skipping chip creation.")
        return chips, rows, cols
    for row in trange(overlap, height - overlap, tile_size, desc='Making chips', leave=False):
        for col in range(overlap, width - overlap, tile_size):
            image = data[:, (row - overlap):(row - overlap + padded_size),
                         (col - overlap):(col - overlap + padded_size)]
            if image.size == 0:
                continue
            image[image == 65535] = 255  # Replace 65535 with 255
            down_pad = max(0, padded_size - image.shape[1])
            right_pad = max(0, padded_size - image.shape[2])
            image = np.pad(image, ((0, 0), (0, down_pad), (0, right_pad)))
            chips.append(image)
            rows.append(row)
            cols.append(col)
    if chips:
        chips = np.stack(chips)
    return chips, rows, cols

def mosaic_logic(existing, new_pred):
    """
    existing: 2D array read from mosaic (values in {0,1,255})
    new_pred: 2D array from the TIF's prediction (values in {0,1,255})

    Returns a merged array that:
       - Overwrites existing=255 with new_pred if new_pred != 255
       - Among valid {0,1} combos, takes the maximum
    """
    merged = existing.copy()

    # 1) Overwrite mosaic 255 with new_pred where new_pred != 255
    mask_nodata_in_dest = (merged == 255) & (new_pred != 255)
    merged[mask_nodata_in_dest] = new_pred[mask_nodata_in_dest]

    # 2) Where both are valid (0 or 1), take the max
    mask_both_valid = (merged != 255) & (new_pred != 255)
    merged[mask_both_valid] = np.maximum(merged[mask_both_valid], new_pred[mask_both_valid])

    return merged

def safe_copy(dest, row, col, src):
    h = min(dest.shape[0] - row, src.shape[2])
    w = min(dest.shape[1] - col, src.shape[3])
    if h <= 0 or w <= 0:
        return

    chunk_dest = dest[row:row + h, col:col + w]
    chunk_src = src[0, 0, :h, :w]  # shape: (h, w)

    # 1) Overwrite where dest=255 but src != 255
    mask_nodata_in_dest = (chunk_dest == 255) & (chunk_src != 255)
    chunk_dest[mask_nodata_in_dest] = chunk_src[mask_nodata_in_dest]

    # 2) Now, for places where both are in {0,1}, take the max
    #    i.e. if dest=0, src=1 => result=1; 
    #    if dest=1, src=0 => keep 1
    mask_both_valid = (chunk_dest != 255) & (chunk_src != 255)
    chunk_dest[mask_both_valid] = np.maximum(
        chunk_dest[mask_both_valid],
        chunk_src[mask_both_valid]
    )

def max_min_normalization_add_ndvi(data):
    ndvi = (data[3] - data[0]) / (data[3] + data[0] + 1e-20)
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and infs
    scaled_ndvi = ((ndvi + 1) * 127.5).astype(np.uint8)
    matched_bands = data.transpose(1, 2, 0)  # Reshape to (height, width, bands)
    uint8_ndvi_expanded = scaled_ndvi[:, :, np.newaxis]
    matched_bands = np.concatenate((matched_bands, uint8_ndvi_expanded), axis=2)
    num_bands = matched_bands.shape[2]
    min_max_ranges = [(0, 222), (0, 221), (0, 224), (0, 216), (0, 255)]
    normalized_bands = []
    for i in range(num_bands):
        band_data = matched_bands[:, :, i]
        min_val, max_val = min_max_ranges[i]
        normalized_band = ((band_data - min_val) / (max_val - min_val)) * 255
        normalized_band = np.clip(normalized_band, 0, 255).astype(np.uint8)
        normalized_bands.append(normalized_band)
    normalized_bands = np.stack(normalized_bands, axis=2)
    nodata_mask = np.all(data == 0, axis=0)
    normalized_bands[nodata_mask] = 255
    normalized_bands = normalized_bands.transpose(2, 0, 1)  # Reshape back to (bands, height, width)
    return normalized_bands

def process_tiles(model, data, tile_size, overlap):
    """Process tiles using the model and return predictions."""
    height, width = data.shape[1], data.shape[2]
    chips, rows, cols = make_chips(data, tile_size, overlap)

    num_chips = len(chips)
    pred = np.zeros((height, width), dtype='uint8')

    for i in tqdm(range(num_chips), desc="Processing chips", leave=False):
        chip = chips[i].astype(np.float32) / 255.0
        chip_tensor = torch.tensor(chip).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = model(chip_tensor)
            probabilities = torch.sigmoid(logits)
            preds = (probabilities > 0.65).float()
        pred_chip = preds.cpu().numpy().astype('uint8')

        # Handle nodata areas
        mask = np.all(chip == 0, axis=0)[np.newaxis, np.newaxis, :, :]
        pred_chip[mask] = 255

        pred_chip_crop = pred_chip[:, :, overlap:-overlap, overlap:-overlap]
        row = rows[i]
        col = cols[i]
        safe_copy(pred, row, col, pred_chip_crop)

        # Free up GPU memory
        del chip_tensor, logits, probabilities, preds
        torch.cuda.empty_cache()

    return pred

def process_file(model, tiff_path, tile_size, overlap, year, reference_dict, out_raster):
    """Process a single TIFF file and write the result into the output raster."""
    tiff_path_s3 = f"s3://naip-analytic/{tiff_path}"

    with rasterio.Env(AWS_REQUEST_PAYER='requester'):
        with rasterio.open(tiff_path_s3, "r") as src:
            data = src.read([1, 2, 3, 4])
            src_transform = src.transform
            src_crs = src.crs
            src_bounds = src.bounds

    # Reproject data if CRS does not match
    if src_crs != out_raster.crs:
        logging.info(f"Reprojecting {tiff_path} from {src_crs} to {out_raster.crs}")
        transform, width, height = calculate_default_transform(
            src_crs, out_raster.crs, src.width, src.height, *src.bounds)
        data_reprojected = np.zeros((4, height, width), dtype=data.dtype)
        reproject(
            source=data,
            destination=data_reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=out_raster.crs,
            resampling=Resampling.nearest,
            src_nodata=65535,
            dst_nodata=255
        )
        data = data_reprojected
        src_transform = transform
        src_bounds = rasterio.coords.BoundingBox(
            left=src_transform.c,
            bottom=src_transform.f + src_transform.e * height,
            right=src_transform.c + src_transform.a * width,
            top=src_transform.f)
    # else: same CRS, continue without reprojection

    # === Normalization and NDVI ===
    data = max_min_normalization_add_ndvi(data)

    # === Optional histogram matching ===
    if reference_dict is not None and len(reference_dict) > 0:
        try:
            logging.info(f"Applying histogram matching for {tiff_path}")
            # You could pick one reference file, or match by year if you extend logic
            ref_path = reference_dict[0]
            data = perform_histogram_matching(data, ref_path)
        except Exception as e:
            logging.warning(f"Skipping histogram matching for {tiff_path}: {e}")

    # === Model inference ===
    prediction = process_tiles(model, data, tile_size, overlap)
    if prediction is None or prediction.size == 0:
        return

    # === Mosaic integration ===
    row_off, col_off = out_raster.index(src_bounds.left, src_bounds.top)
    window = rasterio.windows.Window(col_off, row_off, prediction.shape[1], prediction.shape[0])

    existing_mosaic = out_raster.read(1, window=window)
    merged = mosaic_logic(existing_mosaic, prediction)
    out_raster.write(merged, window=window, indexes=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', required=True, help='Path to text file with list of TIFF file URLs')
    parser.add_argument('--reference_file', required=False, default=None, help='Path to text file with 2020 reference TIFF file URLs (optional)')
    parser.add_argument('--hist_match', action='store_true',help='Enable histogram matching using reference imagery')
    #parser.add_argument('--reference_file', required=True, help='Path to text file with 2020 reference TIFF file URLs')
    parser.add_argument('--output_raster', required=True, help='Path to the output raster file to be created')
    parser.add_argument('--model_checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--tile_size', type=int, default=2048, help='Tile size for processing')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap for tiles')
    parser.add_argument('--year', type=int, required=True, help='Year of the NAIP imagery (used for histogram matching)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_raster), exist_ok=True)

    # Configure logging to show INFO, WARNING, and ERROR messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    if args.hist_match and args.reference_file:
        with open(args.reference_file, 'r') as ref_file:
            reference_paths = [line.strip() for line in ref_file.readlines()]
    else:
        reference_paths = None
    #with open(args.reference_file, 'r') as ref_file:
        #reference_paths = [line.strip() for line in ref_file.readlines()]

    with open(args.text_file, 'r') as f:
        tiff_paths = [line.strip() for line in f.readlines()]

    # Collect metadata from all input TIFF files
    bounds_list = []
    crs_list = []
    with rasterio.Env(AWS_REQUEST_PAYER='requester'):
        for tiff_path in tqdm(tiff_paths, desc='Collecting metadata', leave=False):
            tiff_path_s3 = f"s3://naip-analytic/{tiff_path}"
            with rasterio.open(tiff_path_s3) as src:
                bounds_list.append(src.bounds)
                crs_list.append(src.crs)
                # Get resolution from the first file
                if 'resolution' not in locals():
                    resolution = (src.res[0], src.res[1])  # (pixel width, pixel height)

    # Print CRSs for debugging
    #for i, crs in enumerate(crs_list):
     #   print(f"Raster {i} CRS: {crs}")

    # Verify that all CRSs are the same
    if not all(crs == crs_list[0] for crs in crs_list):
        logging.warning("Not all input rasters have the same CRS. Reprojecting rasters to match output CRS.")
        # Set the output CRS to the first raster's CRS
        crs = crs_list[0]
    else:
        crs = crs_list[0]

    # Calculate the overall bounds
    min_x = min([b.left for b in bounds_list])
    max_x = max([b.right for b in bounds_list])
    min_y = min([b.bottom for b in bounds_list])
    max_y = max([b.top for b in bounds_list])

    # Calculate the dimensions of the output raster
    width = int((max_x - min_x) / resolution[0])
    height = int((max_y - min_y) / abs(resolution[1]))

    output_meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 255,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': rasterio.transform.from_origin(min_x, max_y, resolution[0], resolution[1]),
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'BIGTIFF': 'YES'  # Enable BigTIFF for large output files
        }

    # Open the output raster for writing
    with rasterio.open(args.output_raster, 'w', **output_meta) as out_raster:
        pass  # Close immediately after creating the file with the desired metadata

    # Reopen for read/write operations
    with rasterio.open(args.output_raster, 'r+') as out_raster:
        # Load the model once
        model = CanopyModel.load_from_checkpoint(args.model_checkpoint)
        model.eval()
        model.cuda()

        # Process files sequentially
        for tiff_path in tqdm(tiff_paths, desc='Processing TIFF files'):
            try:
                process_file(model, tiff_path, args.tile_size, args.overlap, args.year, reference_paths, out_raster)
            except Exception as e:
                logging.error(f"Error processing {tiff_path}: {e}")
                logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
