import os
import random
import argparse
import rasterio
from collections import defaultdict
import warnings
import numpy as np

print("Script running...")

# Define area goals
area_goals = {
    "inland_empire": 0.2123,
    "inland_valleys": 0.2754,
    "interior_west": 0.0346,
    "northern_california_coast": 0.164,
    "southern_california_coast": 0.2257,
    "southwest_desert": 0.088
}

# Step 1: Traverse through the directory structure to find mask files of the right size
def find_mask_files(root_dir):
    mask_files = defaultdict(list)
    for region_folder in os.listdir(root_dir):
        mask_dir = os.path.join(root_dir, region_folder, "gridded_masks")  # Path to masks
        if os.path.isdir(mask_dir):
            for file in os.listdir(mask_dir):
                if file.endswith(".tif"):
                    file_path = os.path.join(mask_dir, file)
                    #mask_files[region_folder].append(file_path)
                    if is_right_size(file_path):
                        mask_files[region_folder].append(file_path)
    return mask_files


# Function to check if mask file is of the right size
def is_right_size(file_path):
    with rasterio.open(file_path) as src:
        if src.width != 448 or src.height != 448:
            return False  # Ensure dimensions are correct
        
        # Read the first band of the mask
        data = src.read(1)  
        
        if np.all(data == 255):
            print(f"Skipping {file_path} as it contains only no-data values.")
            return False  # All data are 'no-data' values
        
    return True

def select_files_by_area(mask_files, area_goals):
    selected_files = defaultdict(list)
    max_area_region = max(area_goals, key=area_goals.get)  # Find the region with the highest area goal
    max_area_goal = area_goals[max_area_region]  # Get the area goal of the region with the highest goal

    for region, files in mask_files.items():
        total_files = len(files)  # Count the number of mask files
        if region == max_area_region:
            goal_area = total_files * 448 * 448  # Use all available data for the region with the highest goal
        else:
            goal_area = total_files * 448 * 448 * area_goals[region] / max_area_goal  # Allocate data proportionally

        current_area = 0
        for file in files:
            if current_area < goal_area:
                selected_files[region].append(file)
                current_area += 448 * 448  # Add the area of the current mask
            else:
                break
        
        # Check if selected files are not enough to meet the goal
        if current_area < goal_area:
            warnings.warn(f"Not enough data available to meet the area goal for {region}. Adjusting goal.")
            # You could choose to adjust the goal area here or handle it based on your requirements

    return selected_files

# Step 3: Split the selected files into train, test, and validation sets
def split_data(selected_files, train_ratio, test_ratio, val_ratio):
    train_data, test_data, val_data = [], [], []

    for region, files in selected_files.items():
        random.shuffle(files)  # Ensure random distribution
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        test_end = train_end + int(total_files * test_ratio)

        train_data.extend(files[:train_end])
        test_data.extend(files[train_end:test_end])
        val_data.extend(files[test_end:])

    return train_data, test_data, val_data

def main(root_dir, train_ratio, test_ratio, val_ratio, out_dir):
    mask_files = find_mask_files(root_dir)
    print("Found correctly sized files.")
    selected_files = select_files_by_area(mask_files, area_goals)
    print("Selected files by area goals.")
    train_data, test_data, val_data = split_data(selected_files, train_ratio, test_ratio, val_ratio)

    # Determine output directory
    output_dir = args.out_dir

        # Write train, test, and validation file paths to text files in the specified output directory
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_data))
    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_data))
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process mask files and split them into train, test, and validation sets.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing training data.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to be used for training.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of data to be used for testing.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of data to be used for validation.")
    parser.add_argument("--out_dir", required=True, help="Directory to store output text files. Usually within run_data in the repo.")
    args = parser.parse_args()

    main(args.root_dir, args.train_ratio, args.test_ratio, args.val_ratio, args.out_dir)

