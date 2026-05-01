import os
import argparse
import subprocess
import geopandas as gpd
import pandas as pd
from osgeo import gdal

parser = argparse.ArgumentParser(description="Build NAIP VRTs from the USDA S3 bucket.")
parser.add_argument('--output_dir', type=str, default='./naip_vrts', help='Directory to save VRT files')
parser.add_argument('--shapefile_dir', type=str, default='./naip_shapefiles', help='Directory to save downloaded NAIP index shapefiles')
args = parser.parse_args()

output_dir = args.output_dir
shapefile_dir = args.shapefile_dir

# Create the output and shapefile directories if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(shapefile_dir):
    os.makedirs(shapefile_dir)

# Ensure AWS requester pays is set and the correct region is used
os.environ['AWS_REQUEST_PAYER'] = 'requester'
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

def print_env_vars():
    print("AWS_ACCESS_KEY_ID:", os.getenv('AWS_ACCESS_KEY_ID'))
    print("AWS_SECRET_ACCESS_KEY:", os.getenv('AWS_SECRET_ACCESS_KEY'))
    print("AWS_REQUEST_PAYER:", os.getenv('AWS_REQUEST_PAYER'))
    print("AWS_DEFAULT_REGION:", os.getenv('AWS_DEFAULT_REGION'))

print("Environment Variables:")
print_env_vars()

index_urls = {
    2016: 's3://naip-analytic/ca/2016/60cm/index/naip_3_16_1_1_ca',
    2018: 's3://naip-analytic/ca/2018/60cm/index/NAIP_18_CA',
    2020: 's3://naip-analytic/ca/2020/60cm/index/NAIP_20_CA',
    2022: 's3://naip-analytic/ca/2022/60cm/index/CA_NAIP22_QQ'
}

extensions = ['.shp', '.shx', '.dbf', '.prj']

path_fns = {
    2016: lambda row: f'/vsis3/naip-analytic/ca/2016/60cm/rgbir_cog/{row["USGSID"][:-2]}/{row["s3"]}',
    2018: lambda row: f'/vsis3/naip-analytic/ca/2018/60cm/rgbir_cog/{str(row["USGSID"])[:-2]}/{row["s3"]}',
    2020: lambda row: f'/vsis3/naip-analytic/ca/2020/60cm/rgbir_cog/{str(row["USGSID"])[:-2]}/{row["s3"]}',
    2022: lambda row: f'/vsis3/naip-analytic/ca/2022/60cm/rgbir_cog/{str(row["USGSID"])[:-2]}/{row["s3"]}'
}

def set_gdal_config():
    gdal.SetConfigOption('AWS_REQUEST_PAYER', 'requester')
    gdal.SetConfigOption('AWS_DEFAULT_REGION', 'us-west-2')
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', os.getenv('AWS_ACCESS_KEY_ID'))
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', os.getenv('AWS_SECRET_ACCESS_KEY'))

set_gdal_config()

for year in [2016, 2018, 2020, 2022]:
    print(f'Processing year {year}')

    print(f'Listing files in bucket')
    try:
        res = subprocess.check_output(['aws', 's3', 'ls', '--request-payer', 'requester', '--recursive', f's3://naip-analytic/ca/{year}/60cm/rgbir_cog'])
        lines = res.split(b'\n')[:-1]
        paths = [line.decode('utf-8').split()[-1].split('/')[-1] for line in lines]
        s3df = pd.DataFrame({'s3': paths})
        s3df['s3short'] = s3df['s3'].str[:19]
    except subprocess.CalledProcessError as e:
        print("Error listing files in bucket:", e.output.decode())
        continue

    print('Downloading shapefile')
    shapefile_base_url = index_urls[year]
    shapefile_local_path = os.path.join(shapefile_dir, f'naip_{year}')
    for ext in extensions:
        try:
            subprocess.check_call(['aws', 's3', 'cp', f'{shapefile_base_url}{ext}', f'{shapefile_local_path}{ext}', '--request-payer', 'requester'])
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {ext} for year {year}:", e.output.decode())
            continue

    print('Reading shapefile')
    try:
        gdf = gpd.read_file(f'{shapefile_local_path}.shp')
    except Exception as e:
        print("Error reading shapefile:", e)
        continue

    print('Making S3 paths')
    gdf['indexshort'] = gdf['FileName'].str[:19]
    
    gdf = gdf.merge(s3df, how='inner', left_on='indexshort', right_on='s3short')
    gdf['s3path'] = gdf.apply(path_fns[year], axis=1)

    print('Building VRTs')
    for utm in [10, 11]:
        urls = list(gdf[gdf['UTM'] == utm]['s3path'])
        output_vrt = os.path.join(output_dir, f'naip_{year}_269{utm}.vrt')
        try:
            set_gdal_config()  # Ensure GDAL config is set before accessing the files
            gdal.BuildVRT(output_vrt, urls).FlushCache()
            print(f'VRT file written to: {output_vrt}')
        except Exception as e:
            print(f"Error building VRT for UTM {utm}:", e)

