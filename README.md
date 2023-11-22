# Local Masking Meets Progressive Freezing: Efficient Vision Transformers
*NOTE: Repo is under development.*

## Introduction
This repository contains the implementation of our novel approach for self-supervised learning in Vision Transformers (ViTs), as described in our paper "Local Masking Meets Progressive Freezing: Crafting Efficient Vision Transformers for Self-Supervised Learning". Our method introduces Local Masked Image Modeling (LocalMIM) and a progressive layer freezing mechanism to enhance the training efficiency of ViTs.

## Method Overview
Our approach combines the benefits of local masking, which allows for multi-scale reconstruction, with progressive layer freezing inspired by FreezeOut. This technique enables significant computational savings during training without compromising the model's accuracy.

### LocalMIM
LocalMIM involves partitioning the input image into regions and using these for reconstruction at different scales. This aids the model in capturing fine-grained details as well as broader contextual information.

### Progressive Layer Freezing
We introduce a layer-wise cosine annealing schedule for learning rates, progressively freezing the layers and shifting them to inference mode to save on computation.

## Key Results
Our method achieved a reduction in training time by approximately 12.5% with only a 0.6% decrease in top-1 accuracy, demonstrating the efficiency of our approach.

## Installation

Instructions for setting up the environment and installing required packages.

```bash
# Clone the repository
git clone https://github.com/utkutpcgl/freezeout_localmim_rho

# Navigate to the repository directory
cd freezeout_localmim_rho

# Install the dependencies
pip install -r requirements.txt
```

TODO put seperator line here

# Re-produce Results
## Dataset arrangement
Download the IN1K dataset and the development kit they provide. Arange the dataset in this format:`<ROOT_PATH>/train/image_folders/` and `<ROOT_PATH>/val/image_folders/`. Here `image_folders` should contain all class folders with images in them.
Then update the script `create_ilsvrc2012_val_folders.py` such that the variables point to the target paths:
- `src_dir` to `<ROOT_PATH>/val/image_folders/` 
- `label_file` to `ILSVRC2012_validation_ground_truth.txt` 
- `ilsvrc_mat_file` to `ILSVRC2012_val_info/meta.mat`
Run the script `create_ilsvrc2012_val_folders.py` to arrange the validation folder (will be necessary for fine-tuning.).

*Optional (data_path can be given as cli arguments also):*
Update the argument `data_path` in the train scripts `run_pretrain.py` to point to the `<ROOT_PATH>/train/image_folders/`.
Update the argument `data_path` in the train scripts `run_finetune.py` to point to the `<ROOT_PATH>`.


## Training and Evaluation
The train settings of LocalMIM [] was used. 

### Pre-training
Below are pre-training commands with the hyperparameters adjusted for ViT-B as in LocalMIM. Settings are appropriate for 100 epochs, they should be modified for other epoch schemes. The scaling method and t_0 can be adjusted based on preference (below is our default setting.).
*Note:* Currently training is DDP based, hence, regular single gpu training is not supported. Also, modify `accum_iter` to match the batch size for an iteration (`accum_iter`*`world_size` should equal to 8).

*With freezeout trains:*
- 8 GPU train command *with freezeout* (in the `ViT` folder):
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 8 --accum_iter 1 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir full_pretrain_out_freezeout_cubic_t0_8_fast --log_dir full_pretrain_out_freezeout_cubic_t0_8_fast \
--how_scale cubic --t_0 0.8
```
- 1 GPU train command *with freezeout* (in the `ViT` folder):
```bash
bash record.sh CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir full_pretrain_out_freezeout_cubic_t0_8_1gpu --log_dir full_pretrain_out_freezeout_cubic_t0_8_1gpu \
--how_scale cubic --t_0 0.8
```


*Without freezeout trains:*
- 8 GPU train command *without freezeout* (in the `ViT_orig` folder):
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 8 --accum_iter 1 --weight_decay 0.05 \
--model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir full_pretrain_out_fast --log_dir full_pretrain_out_fast
```

#### Details about the hyperparameters
##### Pre-train Configurations:
- Effective batch size is 2048.
- The learning rate is set to 2e-4 (based on repo README).
- accum_iter set to 2 for 4 gpus (normally 8).
- warm-up epoch is set to 10 for 100 epochs and 40 for all other epoch settings.
- Note that min_lr is not set in the repo's command and is not explained in the paper.
- Code uses as many gpus as there are available, hence, set CUDA_VISIBLE_DEVICES. (optional CUDA_VISIBLE_DEVICES=4,5,6,7 )

Their original command to pre-train ViT-B is:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --batch_size 256 --model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 --epochs 1600 --warmup_epochs 40 --blr 2e-4 --weight_decay 0.05 --data_path /path/to/imagenet/ --output_dir /output_dir/
```




### Fine-tuning
Below are fine-tuning commands with the hyperparameters adjusted for ViT-B as in LocalMIM. We fine-tune it for 100 epochs just as in LocalMIM. 
*Note:* Currently training is DDP based, hence, regular single gpu training is not supported. Also, modify `accum_iter` to match the batch size for an iteration (`accum_iter`*`world_size` should equal to 8).
- 8 GPU (in the `ViT` folder):
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29504 run_finetune.py \
--world_size 8 --accum_iter 1 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/freezeout_localmim_rho/ViT/full_pretrain_out/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir full_finetune_out/ --log_dir full_finetune_out
```

### Reference for citation