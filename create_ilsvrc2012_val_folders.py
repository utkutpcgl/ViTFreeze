import os
import shutil
from scipy.io import loadmat

# Source directory where validation images are stored
src_dir = '/raid/utku/datasets/imagenet/classification/val/image_folders'
# File containing validation labels
label_file = '/raid/home_yedek/utku/freezeout_localmim_rho/ILSVRC2012_validation_ground_truth.txt'
ilsvrc_mat_file = '/raid/home_yedek/utku/freezeout_localmim_rho/ILSVRC2012_val_info/meta.mat'


def create_ilsvrc_class_idx_dict(ilsvrc_mat_file):
    meta = loadmat(ilsvrc_mat_file)
    synsets = meta['synsets']
    # Initialize an empty dictionary to store the mapping
    class_idx_to_code_dict = {}

    # Iterate through the synsets array to populate the dictionary
    for entry in synsets:
        ilsvrc_id = entry[0][0][0][0]  # Extract the ILSVRC2012_ID
        wnid = entry[0][1][0]  # Extract the WNID (class code)
        class_idx_to_code_dict[ilsvrc_id] = wnid
    return class_idx_to_code_dict


def move_images_to_folders(label_file, src_dir, class_idx_to_code_dict):
    # Read and sort the labels from the file
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Iterate through the sorted images and labels
    for i, label in enumerate(labels):
        image_name = f"ILSVRC2012_val_{i+1:08d}.JPEG"  
        src_path = os.path.join(src_dir, image_name)

        # Get the corresponding folder name
        target_folder_name = class_idx_to_code_dict[int(label)]

        # Create target directory if it doesn't exist
        target_dir = os.path.join(src_dir, target_folder_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Destination path
        dest_path = os.path.join(target_dir, image_name)
        
        # Move the image
        shutil.move(src_path, dest_path)

if __name__ == "__main__":
    class_idx_to_code_dict = create_ilsvrc_class_idx_dict(ilsvrc_mat_file)
    move_images_to_folders(label_file, src_dir, class_idx_to_code_dict)