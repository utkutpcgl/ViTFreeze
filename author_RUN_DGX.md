# DETAILS

## Running Experiments

### Pre-training:

Explained in the repos README.md as this, to pre-train ViT-B:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --batch_size 256 --model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 --epochs 1600 --warmup_epochs 40 --blr 2e-4 --weight_decay 0.05 --data_path /path/to/imagenet/ --output_dir /output_dir/
```

#### Train Configurations:
- Effective batch size is 2048
- The learning rate is set to 2e-4 (based on repo README)
- accum_iter set to 2 for 4 gpus (normally 8).
- warm-up epoch is set to 10 for 100 epochs and 40 for all other epoch settings.
- Note that min_lr is not set in the repo's command and is not explained in the paper.
- Code uses as many gpus as there are available, hence, set CUDA_VISIBLE_DEVICES. (optional CUDA_VISIBLE_DEVICES=4,5,6,7 )

*Freezeout Pretrain for 100 epochs:*
```bash
bash record.sh CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 4 --accum_iter 2 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/full_pretrain_out_freezeout_cubic_t0_85 --log_dir full_pretrain_out_freezeout_cubic_t0_85 \
--how_scale cubic --t_0 0.85
```

- 8 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 8 --accum_iter 1 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/full_pretrain_out_freezeout_cubic_t0_8_fast --log_dir full_pretrain_out_freezeout_cubic_t0_8_fast \
--how_scale cubic --t_0 0.8
```

- 1 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29505  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_save5 --log_dir bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_save5 \
--how_scale cubic --t_0 0.8
```

- NON LAYER WISE AND NOT SCALE LR
```bash
bash record.sh CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_1234_loss_scaler --log_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_1234_loss_scaler \
--how_scale cubic --t_0 0.8 \
--non_layerwise_lr

bash record.sh CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29505  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_1overk_loss_scaler_all_stages --log_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_1overk_loss_scaler_all_stages \
--how_scale cubic --t_0 0.8 \
--non_layerwise_lr --all_stages


bash record.sh CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29507  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_all_stages --log_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_all_stages \
--how_scale cubic --t_0 0.8 \
--non_layerwise_lr --all_stages
```

- DONT FREEZE PATCH EMBED NON LAYER WISE AND NOT SCALE LR
```bash
bash record.sh CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29502  run_pretrain.py \
--epochs 100 --batch_size 512 --warmup_epochs 10 \
--blr 2e-4 --world_size 2 --accum_iter 2 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_dont_freeze_pe --log_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_dont_freeze_pe \
--how_scale cubic --t_0 0.8 \
--non_layerwise_lr --dont_freeze_pe

bash record.sh CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29505  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_1overk_loss_scaler_all_stages --log_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_1overk_loss_scaler_all_stages \
--how_scale cubic --t_0 0.8 \
--non_layerwise_lr --all_stages


bash record.sh CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29507  run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_all_stages --log_dir pretrain/non_scale_layerwise/freezeout_cubic_t0_8_all_stages \
--how_scale cubic --t_0 0.8 \
--non_layerwise_lr --all_stages
```

- DEBUG TRAIN
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 run_pretrain.py \
--epochs 9 --batch_size 32 --warmup_epochs 1 \
--blr 2e-4 --world_size 2 --accum_iter 1 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/demo_dataset \
--output_dir pretrain/debug --log_dir debug --debug \
--how_scale cubic --t_0 0.8
```

- DEBUG TRIALS
```bash
bash record.sh CUDA_VISIBLE_DEVICES=3,4,5,6  OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29503  run_pretrain.py \
--epochs 100 --batch_size 512 --warmup_epochs 10 \
--blr 2e-4 --world_size 4 --accum_iter 1 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1 --log_dir pretrain/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1 \
--how_scale cubic --t_0 1.0 \
--non_layerwise_lr

bash record.sh CUDA_VISIBLE_DEVICES=0,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29506  run_pretrain.py \
--epochs 100 --batch_size 512 --warmup_epochs 10 \
--blr 2e-4 --world_size 2 --accum_iter 2 --model MIM_vit_base_patch16 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1_without_remove_freezeout_layers_no_find_up --log_dir pretrain/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1_without_remove_freezeout_layers_no_find_up \
--how_scale cubic --t_0 1.0 \
--non_layerwise_lr
```


*Regular Pretrain for 100 epochs:*

```bash
bash record.sh  CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29502 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 4 --accum_iter 2 --weight_decay 0.05 \
--model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/full_pretrain_out --log_dir full_pretrain_out
```

- 8 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 8 --accum_iter 1 --weight_decay 0.05 \
--model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/full_pretrain_out_fast --log_dir full_pretrain_out_fast
```

- 1 GPU
```bash
bash record.sh CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 run_pretrain.py \
--epochs 100 --batch_size 256 --warmup_epochs 10 \
--blr 2e-4 --world_size 1 --accum_iter 8 --weight_decay 0.05 \
--model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 \
--data_path /raid/utku/datasets/imagenet/classification/train/image_folders \
--output_dir pretrain/full_pretrain_out_1gpu --log_dir full_pretrain_out_1gpu
```

### Fine Tuning:

Explained in the repos README.md as this, to finetune ViT-B:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py --batch_size 128 --model vit_base_patch16 --finetune /path/to/checkpoint.pth --epochs 100 --warmup_epochs 20 --lr 2e-3 --min_lr 1e-5 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path /path/to/imagenet/ --output_dir /output_dir/
```

#### Training configurations
- Effective batch size is 1024.
- For 100-epoch pre-trained model, we set lr=4e-3, layer_decay=0.75 and min_lr=1e-6 

*Finetune Freezeout-100 epochs pre-trained model:*
```bash
bash record.sh CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 run_finetune.py \
--world_size 4 --accum_iter 2 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/full_pretrain_out_freezeout_cubic_t0_8_fast/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_finetune_out_freezeout_cubic_t0_8_fast/ --log_dir finetune/full_finetune_out_freezeout_cubic_t0_8_fast
```

- 8 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 run_finetune.py \
--world_size 8 --accum_iter 1 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/full_pretrain_out_freezeout_cubic_t0_8_fast/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_finetune_out_freezeout_cubic_t0_8_fast/ --log_dir finetune/full_finetune_out_freezeout_cubic_t0_8_fast
```

-- 2 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29507 run_finetune.py \
--world_size 2 --accum_iter 2 \
--batch_size 256 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/non_scale_layerwise_fixed_freezeout_code_orig_finetune/freezeout_cubic_t0_1_without_remove_freezeout_layers_checkpoint-99 --log_dir finetune/non_scale_layerwise_fixed_freezeout_code_orig_finetune/freezeout_cubic_t0_1_without_remove_freezeout_layers_checkpoint-99


bash record.sh CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 run_finetune.py \
--world_size 2 --accum_iter 4 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/full_pretrain_out_freezeout_cubic_t0_8_fast/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_finetune_out_freezeout_cubic_t0_8_fast/ --log_dir finetune/full_finetune_out_freezeout_cubic_t0_8_fast


bash record.sh CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 run_finetune.py \
--world_size 4 --accum_iter 1 \
--batch_size 256 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1_checkpoint-99 --log_dir finetune/non_scale_layerwise_fixed_cls_token_wd/freezeout_cubic_t0_1_checkpoint-99


```

-- 4 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 run_finetune.py \
--world_size 4 --accum_iter 1 \
--batch_size 256 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers_checkpoint-99 --log_dir finetune/non_scale_layerwise_fixed_freezeout_code/freezeout_cubic_t0_1_without_remove_freezeout_layers_checkpoint-99
```

-- 1 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 run_finetune.py \
--world_size 1 --accum_iter 8 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT_orig/full_pretrain_out_slow/checkpoint-80.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_pretrain_out_slow/checkpoint-80 --log_dir finetune/full_pretrain_out_slow/checkpoint-80

bash record.sh CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 run_finetune.py \
--world_size 1 --accum_iter 8 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT_orig/full_pretrain_out_slow/checkpoint-70.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_pretrain_out_slow/checkpoint-70 --log_dir finetune/full_pretrain_out_slow/checkpoint-70

bash record.sh CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 run_finetune.py \
--world_size 1 --accum_iter 8 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT_orig/full_pretrain_out_slow/checkpoint-60.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_pretrain_out_slow/checkpoint-60 --log_dir finetune/full_pretrain_out_slow/checkpoint-60

bash record.sh CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29506 run_finetune.py \
--world_size 1 --accum_iter 8 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT_orig/full_pretrain_out_slow/checkpoint-50.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/full_pretrain_out_slow/checkpoint-50 --log_dir finetune/full_pretrain_out_slow/checkpoint-50

bash record.sh CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 run_finetune.py \
--world_size 1 --accum_iter 8 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/utku/ViTFreeze/ViT/pretrain/non_scale_layerwise_failed/freezeout_cubic_t0_8/checkpoint-59.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/non_scale_layerwise/freezeout_cubic_t0_8_checkpoint-59 --log_dir finetune/non_scale_layerwise/freezeout_cubic_t0_8_checkpoint-59

```


-- Resume 1 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29507 run_finetune.py 
--resume /raid/home_yedek/utku/ViTFreeze/ViT/finetune/bench_1/full_finetune_out_freezeout_cubic_t0_85/checkpoint-40.pth \
--world_size 1 --accum_iter 8 \
--batch_size 128 --model vit_base_patch16 \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir finetune/bench_1/full_finetune_out_freezeout_cubic_t0_85 --log_dir finetune/bench_1/full_finetune_out_freezeout_cubic_t0_85
```

*Finetune Regular-100 epochs pre-trained model:*
```bash
bash record.sh CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29504 run_finetune.py \
--world_size 4 --accum_iter 2 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/full_pretrain_out/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir full_finetune_out/ --log_dir full_finetune_out
```

- 8 GPU:
```bash
bash record.sh CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29504 run_finetune.py \
--world_size 8 --accum_iter 1 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/full_pretrain_out/checkpoint-99.pth \
--epochs 100 --warmup_epochs 20 --lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 \
--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir full_finetune_out/ --log_dir full_finetune_out
```



### Linear Probing:
- 2048 batch size * 8 gpu (or accum_iter)
- blr is 0.1 --->  lr = 0.1*2048/256 * 8 = 6.4
- weight_decay is 0.0
- epochs 90
- model vit_base_patch16
- cls_token 
- world_size 1
- accum_iter 8


```bash
bash record.sh CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29506 run_linprobe.py \
--world_size 1 --accum_iter 8 \
--batch_size 2048 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/bench_1/full_pretrain_out_freezeout_cubic_t0_8_fast/checkpoint-99.pth \
--epochs 90 --warmup_epochs 10 --blr 0.1 \
--weight_decay 0.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir linprob_out/bench_1 --log_dir linprob_out/bench_1
```
```bash
bash record.sh CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29507 run_linprobe.py \
--world_size 1 --accum_iter 8 \
--batch_size 2048 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/bench_2/full_pretrain_out_freezeout_cubic_t0_65_1gpu/checkpoint-99.pth \
--epochs 90 --warmup_epochs 10 --blr 0.1 \
--weight_decay 0.0 --dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir linprob_out/bench_2 --log_dir linprob_out/bench_2
```

### k-NN Clasification:
- results should be independent of batch-size 
- model vit_base_patch16
- orig image is 256, and it is cropped to 224
- cls_token 
- world_size 1
- accum_iter 8


```bash
bash record.sh CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 run_knn.py \
--world_size 1 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_not_scale_lr/checkpoint-99.pth \
--dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir knn_out/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_not_scale_lr_checkpoint-99 --log_dir knn_out/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_not_scale_lr_checkpoint-99

bash record.sh CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29506 run_knn.py \
--world_size 1 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT/pretrain/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_save5/checkpoint-99.pth \
--dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir knn_out/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_save5_checkpoint-99 --log_dir knn_out/bench_3/full_pretrain_out_freezeout_cubic_t0_8_1gpu_save5_checkpoint-99

bash record.sh CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 run_knn.py \
--world_size 1 \
--batch_size 128 --model vit_base_patch16 --finetune /raid/home_yedek/utku/ViTFreeze/ViT_orig/full_pretrain_out_1gpu/checkpoint-99.pth \
--dist_eval \
--data_path /raid/utku/datasets/imagenet/classification/ \
--output_dir knn_out/fullpretrain/full_pretrain_out_1gpu_checkpoint-99 --log_dir knn_out/fullpretrain/full_pretrain_out_1gpu_checkpoint-99
```



