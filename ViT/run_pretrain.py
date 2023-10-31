# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.freezeout_utils import create_param_groups, get_param_groups, validate_same_objects, get_freezeout_modules, FREEZEOUT_LAYER_COUNT_VIT_B
import models_mim
from engine_pretrain import train_one_epoch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# NOTE scale function
scale_fn = {'linear':lambda x: x,
            'squared': lambda x: x**2,
            'cubic': lambda x: x**3}


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU (effective batch size is batch_size*accum_iter*ngpus') # 8*256 = 2048 for base model
    parser.add_argument('--epochs', default=5, type=int) # 100 for initial training
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='MIM_vit_base_patch16', type=str, help='Name of model to train') # NOTE was mae_vit_large_patch16
    parser.add_argument('--how_scale', default="cubic", type=str, help='Select how to scale the model.') # NOTE was mae_vit_large_patch16
    parser.add_argument('--t_0', default=0.8, type=float, help='Select freezeout specific t_0 (when to freeze the intial layer)') # NOTE was mae_vit_large_patch16
    parser.add_argument('--input_size', default=224, type=int, help='images input size') # 224 always
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).') # 0.75 always
    parser.add_argument('--hog_nbins', default=9, type=int, help='nbins for HOG feature') # NOTE paper says 18 but here and in the repo it says 9??
    parser.add_argument('--hog_bias', action='store_true', help='hog bias')
    parser.set_defaults(hog_bias=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, help='absolute_lr = base_lr*total_batch_size/256') # 2*10^-4 for base model
    parser.add_argument('--min_lr', type=float, default=1e-6, help='lower lr bound for cyclic schedulers that hit 0') # NOTE the paper provides no information, the repo does not modify it.
    parser.add_argument('--warmup_epochs', type=int, default=0, help='epochs to warmup LR')

    # Dataset parameters
    #TODO update when necessray data_path.
    parser.add_argument('--data_path', default='/raid/utku/datasets/imagenet/classification/train/demo_dataset/', type=str, help='dataset path orig: /raid/utku/datasets/imagenet/classification/train/image_folders \
                        demo: /raid/utku/datasets/imagenet/classification/train/demo_dataset/' )
    parser.add_argument('--output_dir', default='./output/MAE_ViT_B', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default="runs/pretrain/exp", help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False) # TODO make this work for the long
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int) # NOTE this is per gpu.
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def main(args):
    misc.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    assert len(dataset_train) != 0
    print(len(dataset_train))
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # NOTE Def effective batch size and lr (represents initial learning rate).
    eff_batch_size = args.batch_size*args.accum_iter*misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr*eff_batch_size/256
    print("base lr: %.2e" % (args.lr*256/eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # define the model
    model = models_mim.__dict__[args.model](hog_nbins=args.hog_nbins, hog_bias=args.hog_bias, how_scale=args.how_scale, t_0=args.t_0)
    model.to(device)
    lr_scale_fn = scale_fn[model.how_scale] # freezeout spec
    t_0 = model.t_0 # freezeout spec
    num_of_layers = model.cum_layer_index # freezeout spec
    iterations_per_epoch = len(data_loader_train) # NOTE (len(dataset)/batch_size).
    assert iterations_per_epoch != 0
    assert hasattr(model.patch_embed, "layer_index")
    freezeout_module_level_specifier_count = 0
    for module in model.modules():
        if hasattr(module,'active'): # freezout specific
            print("Set module with layer_index:", module.layer_index)
            # the ratio to be multiplied with the initial learning rate.
            module.lr_ratio = lr_scale_fn(t_0 + (1 - t_0) * float(module.layer_index) / num_of_layers) # freezout specific
            module.initial_lr = args.lr/module.lr_ratio if model.scale_lr else args.lr # freezout specific
            # NOTE iterations set auto instead of 1000 (so in freezeout), warmup is not included.
            module.max_iteration = (args.epochs-args.warmup_epochs) * iterations_per_epoch * module.lr_ratio # freezout specific, the maximum count a layer will be trained for (after max_iteration it will be frozen), hardcoded 1000 iterations per epoch.
            module.freezeout_module_level_specifier = None # Just a module level specifier to distinguish module freezeout layer levels.
            freezeout_module_level_specifier_count+=1
    print("freezeout_module_level_specifier_count: ", freezeout_module_level_specifier_count)
    assert freezeout_module_level_specifier_count == FREEZEOUT_LAYER_COUNT_VIT_B
    model_without_ddp = model


    # problematic_params = {}

    # # Find parameters at the specified indices
    # target_indices = {147, 148, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,180,181}

    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     if idx in target_indices:
    #         problematic_params[idx] = (name, param)
    # print(problematic_params)
    # raise Exception

    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # NOTE I want only parameters of the encoder layer to have these freezeout specific param_groups.
    # Default: param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay) 
    # Default optimizer: optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer_param_groups = create_param_groups(model_without_ddp,log_writer=log_writer)
    # NOTE parameters groups set with explicit learning_rate (or other params) will ignore the learning rate of AdamW arguments.
    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(0.9, 0.95)) # freezout specific optimizer
    loss_scaler = NativeScaler()

    misc.auto_load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # NOTE initialize the param_groups after auto loading the model and optimizer in case starting from checkpoint
    param_groups = get_param_groups(optimizer, test=False, log_writer=log_writer) # freezout specific
    active_freezeout_modules = get_freezeout_modules(model_without_ddp)
    validate_same_objects(optimizer, param_groups["freezeout"]) # freezout specific assertion

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, param_groups, active_freezeout_modules=active_freezeout_modules, log_writer=log_writer, args=args)
        if args.output_dir and (epoch%50 == 0 or epoch+1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)