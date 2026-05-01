# Data

## Training Data
Training chips (normalized image chips and corresponding canopy masks) are available for download at:
[TODO: add Dryad DOI]

The training chips are expected in the following structure for use with the training scripts:
```
training_chips/
  <climate_zone>/
    gridded_images_normalized/
    gridded_masks/
```

## NAIP Imagery
NAIP imagery is available via the USDA requester-pays S3 bucket:
`s3://naip-analytic/`

You will need AWS credentials and to set `AWS_REQUEST_PAYER=requester` in your environment.
VRT files for each UTM zone and year can be built using `make_aws_vrts.py`.

## accuracy_assessment/
Shapefiles for the accuracy assessment point samples used to validate the canopy cover maps.

Each row is one reference point used for accuracy assessment and error-adjusted area estimation following Olofsson et al. (2014).

| Column | Description |
|--------|-------------|
| `urban` | Whether the point falls in an urban (1) or non-urban (0) area |
| `cz` | California climate zone the point falls in (Southwest Desert, NorCal Coast, SoCal Coast, Inland Valleys, Inland Empire, Interior West) |
| `t_16_t` | Reference label for 2016 NAIP imagery (0 = non-canopy, 1 = canopy, NA = ambiguous) |
| `t_18_t` | Reference label for 2018 |
| `t_20_t` | Reference label for 2020 |
| `t_22_t` | Reference label for 2022 |
| `p_2016_all` | Model-predicted class for 2016 (0 = non-canopy, 1 = canopy) |
| `p_2018_all` | Model-predicted class for 2018 |
| `p_2020_all` | Model-predicted class for 2020 |
| `p_2022_all` | Model-predicted class for 2022 |
| `geometry` | Point geometry |

