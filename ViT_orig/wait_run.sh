#!/bin/bash

# Define the remote host and user
REMOTE_HOST="192.168.1.100"
REMOTE_USER="kuartis-dgx1"
PASSWORD="kuartis2012"

TARGET_PIDS=(3972200) # Add your PIDs here

# Create an SSH command with sshpass
SSH_COMMAND="sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no -T ${REMOTE_USER}@${REMOTE_HOST}"

# # Start the SSH session and the while loop on the remote host
# $SSH_COMMAND << 'EOF'
# # Loop through all the PIDs
# for TARGET_PID in "${TARGET_PIDS[@]}"; do
#   while ps -p $TARGET_PID > /dev/null 2>&1; do
#       echo "Process $TARGET_PID is still running. Checking again in 1 seconds..."
#       sleep 1
#   done
#   echo "Process $TARGET_PID has finished."
# done
# EOF

# Start the SSH session and the while loop on the remote host
$SSH_COMMAND << EOF
# Read the PIDs into an array on the remote shell
read -a pid_array <<< "$TARGET_PIDS"
# Loop through all the PIDs
for TARGET_PID in "\${pid_array[@]}"; do
  while ps -p \$TARGET_PID > /dev/null 2>&1; do
      echo "Process \$TARGET_PID is still running. Checking again in 1 seconds..."
      sleep 1
  done
  echo "Process \$TARGET_PID has finished."
done
EOF

echo "All processes have finished, start training."


bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 8 --accum_iter 1 --weight_decay 0.05 \
--model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir full_pretrain_out_fast --log_dir full_pretrain_out_fast