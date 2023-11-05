#!/bin/bash

# Define the source base path for convenience
SOURCE_BASE="/raid/home_yedek/utku/freezeout_localmim_rho/ViT"

# Define the target folder
TARGET_FOLDER="/raid/utku/backup/tez/freezeout"

# Ensure the target folder exists
mkdir -p "$TARGET_FOLDER"

# Copy each directory recursively to the target folder
cp -r "$SOURCE_BASE/full_pretrain_out_freezeout_linear_t0_5_slow_wrong" "$TARGET_FOLDER"
cp -r "$SOURCE_BASE/full_pretrain_out_freezeout_cubic_wrong" "$TARGET_FOLDER"
cp -r "$SOURCE_BASE/full_pretrain_out_freezeout_cubic_t0_8_fast" "$TARGET_FOLDER"
cp -r "$SOURCE_BASE/full_finetune_out_freezeout_cubic_wrong" "$TARGET_FOLDER"
cp -r "$SOURCE_BASE/full_finetune_out_freezeout_cubic_t0_8_fast" "$TARGET_FOLDER"

echo "Copy completed."
