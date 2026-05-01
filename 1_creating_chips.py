
import os
import argparse
import copy
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import mask
from shapely.geometry import box
import numpy as np

def create_directory_structure(output_root_folder, cz):
    cz_mapping = {
        "Southwest Desert": "southwest_desert",
        "Interior West": "interior_west",
        "Inland Empire": "inland_empire",
        "Inland Valleys": "inland_valleys",
        "Northern California Coast": "northern_california_coast",
        "Southern California Coast": "southern_california_coast"
    }
    subfolder = cz_mapping.get(cz, None)
    if not subfolder:
        raise ValueError("Unknown climate zone")
    
    output_folder = os.path.join(output_root_folder, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    for subfolder_name in ["gridded_images", "gridded_masks"]:
        subfolder_path = os.path.join(output_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
    
    return output_folder

def get_climate_zone(climate_zones, geometry, crs):
    climate_zones = climate_zones.to_crs(crs)
    intersections = climate_zones.intersection(geometry)
    max_index = intersections.area.argmax()
    return climate_zones.iloc[max_index]['cz']

def get_grid(raster, size, buffer=2): #there is padding to ensure no odd edge effects in clipping
    transform = raster.transform
    ncols, nrows = raster.meta['width'], raster.meta['height']
    width, height = transform.a, -transform.e  # pixel dimensions

    buffered_size = size + 2 * buffer  # Increase the cell size by the buffer on each side

    grid = []
    for i in range(0, ncols, size):
        for j in range(0, nrows, size):
            # Define grid cell corners in coordinate space with added buffer
            x1 = transform.c + i * transform.a - buffer * width
            y1 = transform.f + j * transform.e + buffer * height
            x2 = x1 + buffered_size * transform.a
            y2 = y1 - buffered_size * transform.e
            grid.append(box(x1, y1, x2, y2))
    
    return grid


def save_raster(image_raster_path, geometry, image_output_path, mask_raster_path=None):
    buffer = 2  # Number of pixels buffer
    target_size = 448
    with rasterio.open(image_raster_path) as image_raster:
        out_image, out_transform = mask.mask(image_raster, [geometry], crop=True)
        if out_image.size == 0:
            print(f"No data to save for {image_output_path} (image data)")
            return

        # Crop to the central target_size
        crop_start = buffer
        crop_end = crop_start + target_size
        out_image = out_image[:, crop_start:crop_end, crop_start:crop_end]

        # Align and pad image
        out_meta = copy.copy(image_raster.meta)
        out_meta.update({
            "driver": "GTiff",
            "height": 448,
            "width": 448,
            "transform": out_transform
        })

        with rasterio.open(image_output_path, 'w', **out_meta) as dst:
            dst.write(out_image)

        if mask_raster_path:
            with rasterio.open(mask_raster_path) as mask_raster:
                out_mask, out_mask_transform = mask.mask(mask_raster, [geometry], crop=True)
                if out_mask.size == 0:
                    print(f"No data to save for mask at {mask_output_path} (mask data)")
                    return

                # Ensure transformation aligns with the image
                out_mask_transform = out_transform

                out_mask = out_mask[:, crop_start:crop_end, crop_start:crop_end]
                mask_output_path = image_output_path.replace("gridded_images", "gridded_masks")
                out_mask_meta = copy.copy(mask_raster.meta)
                out_mask_meta.update({
                    "driver": "GTiff",
                    "height": 448,
                    "width": 448,
                    "transform": out_mask_transform
                })

                with rasterio.open(mask_output_path, 'w', **out_mask_meta) as dst:
                    dst.write(out_mask)



def main():
    print("Script started")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--shapefile-path', type=str, help='Path to the shapefile')
    parser.add_argument('--csv-path', type=str, help='Path to the CSV file')
    parser.add_argument("--input-vrt-folder", type=str, help="Root folder for vrt files.")
    parser.add_argument("--input-canopy-folder", type=str, help="Root folder for canopy tiff files.")
    parser.add_argument("--output-chips-folder", type=str, help="Root folder for storing output chips.")
    parser.add_argument('--size', type=int, default=448, help='Pixel dimensions for the square grid (default: 448)')
    parser.add_argument("--climate-zone-shapefile", type=str, help="Shapefile containing climate zone data.")
    args = parser.parse_args()

    os.environ['AWS_REQUEST_PAYER'] = 'requester'

    places = gpd.read_file(args.shapefile_path)
    data = pd.read_csv(args.csv_path)

    for _, row in data.iterrows():
        city_name = row['city_name']
        folder_name = row['folder_name']
        year = row['naip year']
        utm = row['utm']
        vrt_path = os.path.join(args.input_vrt_folder, f"naip_{year}_{utm}.vrt")
        tiff_path = os.path.join(args.input_canopy_folder, folder_name, "points.laz", "canopy.tif")
        climate_zone_shp = gpd.read_file(args.climate_zone_shapefile)

        with rasterio.open(tiff_path) as src:
            src_crs = src.crs
            grid_polygons = get_grid(src, args.size)
            if not grid_polygons:
                raise ValueError("No grid polygons generated")
            grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=src.crs)
            print(f"{city_name} has {len(grid_gdf)} grid cells total.")

            places_transformed = places.to_crs(src_crs)
            climate_zones_transformed = climate_zone_shp.to_crs(src_crs)

            city_boundary = places_transformed[places_transformed['NAME'] == city_name].geometry.unary_union
            if city_boundary.is_empty:
                raise ValueError(f"No valid geometry for {city_name}.")
            
            climate_zone = get_climate_zone(climate_zones_transformed, city_boundary, src_crs)
            output_folder = create_directory_structure(args.output_chips_folder, climate_zone)
            print(climate_zone)

            cropped_grid = grid_gdf[grid_gdf.within(city_boundary)]
            print(f"{city_name} has {len(cropped_grid)} grid cells within the city boundary.")

            for index, grid_row in cropped_grid.iterrows():
                if not city_boundary.intersects(grid_row.geometry):
                    continue
                image_path = os.path.join(output_folder, "gridded_images", f"{city_name}_{index}.tif")
                save_raster(vrt_path, grid_row.geometry, image_path, mask_raster_path=tiff_path)

if __name__ == '__main__':
    main()
