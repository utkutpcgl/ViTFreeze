# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
#
# 2023.3.3-Changed for building LocalMIM
#          Huawei Technologies Co., Ltd. <foss@huawei.com>

import math
import sys
import torch
import time
import datetime

import util.misc as misc
import util.lr_sched as lr_sched
import util.freezeout_utils as fo_sched

def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, param_groups, active_freezeout_modules, non_layerwise_lr, log_writer=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter # NOTE Accumulate gradient iterations (for increasing the effective batch size under memory constraints, normally is 1.

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    time_consume = 0.
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.synchronize()
        start = time.time()
        # we use a per iteration (instead of per epoch) lr scheduler
        samples = samples.to(device, non_blocking=True) # NOTE this should be faster if placed before adjust_learning_rate_freezeout.
        if data_iter_step % accum_iter == 0:
            # Default was: lr_sched.adjust_learning_rate(optimizer, data_iter_step/len(data_loader)+epoch, args)
            # NOTE lr and attributes have to be set for all models (all ranks.)
            min_active_layer_index = fo_sched.adjust_learning_rate_freezeout(optimizer,  epoch, data_iter_step, param_groups, active_freezeout_modules=active_freezeout_modules, iter_per_epoch=len(data_loader), writer=log_writer, args=args, non_layerwise_lr=non_layerwise_lr) # Freezeout specific

        with torch.cuda.amp.autocast():
            loss = model(samples, min_active_layer_index=min_active_layer_index,mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step+1)%accum_iter==0)
        if (data_iter_step+1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if param_groups is None: # default optimizer debug mode
            lr = optimizer.param_groups[0]["lr"] # Default was this
        else: #freezeout param_groups are set.
            lr = param_groups["non_freezeout"][0]["lr"] # Freezeout specific
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step/len(data_loader)+epoch)*1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        torch.cuda.synchronize()
        time_consume += time.time() - start
    total_time_str = str(datetime.timedelta(seconds=int(time_consume)))
    print('True time {}'.format(total_time_str))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}