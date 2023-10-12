import os
import shutil
from tqdm import tqdm

source_dir = '/raid/utku/datasets/imagenet/classification/train/image_folders'
target_dir = '/raid/utku/datasets/imagenet/classification/train/demo_dataset/'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate over each folder in the source directory
for folder_name in tqdm(os.listdir(source_dir)):
    source_folder_path = os.path.join(source_dir, folder_name)
    target_folder_path = os.path.join(target_dir, folder_name)
    
    # Ensure the current item is indeed a folder
    if os.path.isdir(source_folder_path):
        
        # Get the list of all files in the folder
        files = [f for f in os.listdir(source_folder_path) if os.path.isfile(os.path.join(source_folder_path, f))]
        
        if files:  # Check if the folder has any files
            # Copy the first file to the target directory
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)
            
            source_file = os.path.join(source_folder_path, files[0])
            target_file = os.path.join(target_folder_path, files[0])
            
            shutil.copy2(source_file, target_file)
