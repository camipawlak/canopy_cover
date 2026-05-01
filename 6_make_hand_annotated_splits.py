import os
import argparse
import random

def find_mask_files(root_dir):
    mask_files = {}
    # Only search in the 'masks' subdirectories of each 'cz' directory
    for root, dirs, files in os.walk(root_dir):
        if 'masks' in root.split(os.sep):
            region = os.path.basename(os.path.dirname(root))  # Assume parent directory is the 'cz' directory
            if region not in mask_files:
                mask_files[region] = []
            for file in files:
                if file.endswith(".tif") or file.endswith(".tiff"):  # Check for .tif or .tiff extensions
                    mask_files[region].append(os.path.join(root, file))
    return mask_files

def split_data(selected_files, train_ratio, test_ratio, val_ratio):
    train_data, test_data, val_data = {}, {}, {}
    for region, files in selected_files.items():
        random.shuffle(files)  # Randomize the order of files for a fair split
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        test_end = train_end + int(total_files * test_ratio)

        train_data[region] = files[:train_end]
        test_data[region] = files[train_end:test_end]
        val_data[region] = files[test_end:]
    return train_data, test_data, val_data

def write_data_files(root_dir, data, filename):
    for region, files in data.items():
        output_dir = os.path.join(root_dir, region)  # Adjusted to save directly in the region folder
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        with open(os.path.join(output_dir, filename), "w") as file:
            file.write("\n".join(files))
        print(f"Written {filename} for {region}")


def main(root_dir, train_ratio, test_ratio, val_ratio):
    mask_files = find_mask_files(root_dir)
    print("Found mask files across regions.")
    train_data, test_data, val_data = split_data(mask_files, train_ratio, test_ratio, val_ratio)
    print("Split data into train, test, and validation sets.")

    # Write data to respective files within each region's 'masks' directory
    write_data_files(root_dir, train_data, "train.txt")
    write_data_files(root_dir, test_data, "test.txt")
    write_data_files(root_dir, val_data, "val.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process mask files and split them into train, test, and validation sets.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing training data.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to be used for training.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of data to be used for testing.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of data to be used for validation.")
    args = parser.parse_args()

    main(args.root_dir, args.train_ratio, args.test_ratio, args.val_ratio)
