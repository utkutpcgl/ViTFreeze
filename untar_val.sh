#!/bin/bash

# 1. Delete the folder if it exists
destination_folder="/raid/utku/datasets/imagenet/classification/val/image_folders/"
if [ -d "$destination_folder" ]; then
    rm -rf "$destination_folder"
fi

# 2. Untar the content to the specified folder
tar_file_path="/raid/utku/datasets/imagenet/classification/val/ILSVRC2012_img_val.tar"

# Create the destination folder
mkdir -p "$destination_folder"

# Untar the content
tar -xvf "$tar_file_path" -C "$destination_folder"
