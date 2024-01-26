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


bash record.sh CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 run_finetune.py \
--world_size 4 --accum_iter 1 \
--batch_size 256 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers_checkpoint-99 --log_dir finetune/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers_checkpoint-99