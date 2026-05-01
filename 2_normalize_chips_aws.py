import os
import argparse
import rasterio
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
import numpy as np
from skimage.exposure import match_histograms
import shapely.geometry as sg
import multiprocessing

os.environ['AWS_REQUEST_PAYER'] = 'requester'

# Perform histogram matching
def perform_histogram_matching(image_orig, vrt_path):
    try:
        with rasterio.open(image_orig) as src_orig:
            bounds = src_orig.bounds
            bbox_polygon = sg.box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            with rasterio.open(vrt_path) as src_vrt:
                raster_data_vrt, _ = rasterio.mask.mask(src_vrt, [bbox_polygon], crop=True)
                raster_data_orig = src_orig.read()

                #print(f"Original raster data shape: {raster_data_orig.shape}")
                #print(f"VRT raster data shape: {raster_data_vrt.shape}")
                
                raster_data_orig = reshape_as_image(raster_data_orig)
                raster_data_vrt = reshape_as_image(raster_data_vrt)
                
                matched_image_orig = match_histograms(raster_data_orig, raster_data_vrt, channel_axis=2).astype(np.uint8)
                return matched_image_orig
    except Exception as e:
        print(f"Error in perform_histogram_matching: {e}")
        return None

def max_min_normalization_add_ndvi(path_to_input, path_to_output, vrt_dir):
    try:
        with rasterio.open(path_to_input) as src:
            data = src.read()
            transform = src.transform
            crs = src.crs
            #data = data[:, -448:, -448:] #last 448 pixels because first is na
            #print(f"Input raster data shape: {data.shape}, Bands: {src.count}")
            #print(crs)
            utm_zone = int(str(crs).split('EPSG:')[1].split(',')[0])
            #print(utm_zone)
            if utm_zone == 26911:
                vrt_path = os.path.join(vrt_dir, "naip_2020_26911.vrt")
            elif utm_zone == 26910:
                vrt_path = os.path.join(vrt_dir, "naip_2020_26910.vrt")
            else:
                raise ValueError(f"Unsupported UTM zone: {utm_zone}")
            
            #print(crs)
            #print(vrt_path)
            ndvi = (data[3] - data[0]) / (data[3] + data[0] + 1e-20)
            scaled_ndvi = ((ndvi + 1) * 127.5).astype(np.uint8)
            
            matched_bands = perform_histogram_matching(path_to_input, vrt_path)
            #matched_bands = matched_bands[:, -448:, -448:] #last 448 pixels because first is na
            if matched_bands is None:
                raise ValueError("Histogram matching failed")
            
            #print(f"Matched bands shape: {matched_bands.shape}")
            #print(f"NDVI shape: {scaled_ndvi.shape}")
            
            # Ensure NDVI has the correct shape
            if scaled_ndvi.shape != (448, 448):
                raise ValueError(f"Unexpected NDVI shape: {scaled_ndvi.shape}")

            uint8_ndvi_expanded = scaled_ndvi[:, :, np.newaxis]
            #print(f"NDVI expanded shape: {uint8_ndvi_expanded.shape}")
            
            matched_bands = np.concatenate((matched_bands, uint8_ndvi_expanded), axis=2)
            #print(f"Concatenated bands shape: {matched_bands.shape}")

            num_bands = matched_bands.shape[2]
            min_max_ranges = [(0, 222), (0, 221), (0, 224), (0, 216), (0, 255)]
            normalized_bands = []

            assert len(min_max_ranges) == num_bands, f"Mismatch in number of bands and number of min-max ranges: {num_bands}"

            for i in range(num_bands):
                band_data = matched_bands[:, :, i]
                min_val, max_val = min_max_ranges[i]
                #print(f"Normalizing band {i}, min: {min_val}, max: {max_val}")
                normalized_band = ((band_data - min_val) / (max_val - min_val)) * 255
                normalized_band = normalized_band.astype(np.uint8)
                
                if i == 0:
                    normalized_bands = np.expand_dims(normalized_band, axis=2)
                else:
                    normalized_bands = np.concatenate((normalized_bands, np.expand_dims(normalized_band, axis=2)), axis=2)
            
            
            #print(f"Normalized bands shape: {normalized_bands.shape}")
            
            # Set nodata values to 255
            nodata_mask = np.all(data == 0, axis=0)
            normalized_bands[nodata_mask] = 255
            #normalized_bands = normalized_bands[-448:, -448:, :] #last 448 pixels because first is na
            profile = src.profile
            profile.update(dtype=rasterio.uint8, count=(normalized_bands.shape[2]))
         
            
            with rasterio.open(path_to_output, 'w', **profile) as dst:
                for i in range(normalized_bands.shape[2]):
                    band_data = normalized_bands[:,:,i]
                    dst.write(band_data, i + 1)
    except IndexError as ie:
        print(f"IndexError in max_min_normalization_add_ndvi: {ie}")
        print(f"Error in file: {path_to_input}")
    except Exception as e:
        print(f"Error in max_min_normalization_add_ndvi: {e}")
        print(f"Error in file: {path_to_input}")

def process_file(file_data):
    path_to_input, path_to_output, overwrite, vrt_dir = file_data
    if not overwrite and os.path.exists(path_to_output):
        print(f'Skipping {os.path.basename(path_to_input)} as it already exists.')
    else:
        try:
            max_min_normalization_add_ndvi(path_to_input, path_to_output, vrt_dir)
            print(f'Completed normalization for: {os.path.basename(path_to_input)}')
        except Exception as e:
            print(f"Error processing {os.path.basename(path_to_input)}: {e}")

def process_directory(input_directory, overwrite, vrt_dir):
    tasks = []
    for subfolder in os.listdir(input_directory):
        path_subfolder = os.path.join(input_directory, subfolder)
        if os.path.isdir(path_subfolder):
            gridded_images = os.path.join(path_subfolder, 'gridded_images')
            gridded_images_normalized = os.path.join(path_subfolder, 'gridded_images_normalized')

            if not os.path.exists(gridded_images_normalized):
                os.makedirs(gridded_images_normalized)

            for filename in os.listdir(gridded_images):
                if filename.endswith(".tif"):
                    path_to_input = os.path.join(gridded_images, filename)
                    path_to_output = os.path.join(gridded_images_normalized, filename)
                    tasks.append((path_to_input, path_to_output, overwrite, vrt_dir))

    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, num_cores - 2)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_file, tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and add NDVI to raster images.")
    parser.add_argument('input_directory', type=str, help="Directory containing subfolders with gridded images.")
    parser.add_argument('--vrt_dir', type=str, required=True, help="Directory containing the NAIP 2020 VRT files (naip_2020_26910.vrt and naip_2020_26911.vrt). Build these with make_aws_vrts.py.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing files in the normalized directory.")
    args = parser.parse_args()

    process_directory(args.input_directory, args.overwrite, args.vrt_dir)
