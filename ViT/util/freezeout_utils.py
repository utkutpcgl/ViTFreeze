import numpy as np
from timm.models.vision_transformer import Block # NOTE has internal skip connections.
from torch import nn
import math
import pandas as pd
from functools import reduce


# ---------------------------------- LAYER AWARE ATTRIBUTES
class AttributeAwareModule(nn.Module):
    def named_parameters(self, prefix='', recurse=True):
        gen = super().named_parameters(prefix=prefix, recurse=recurse)
        for elem in gen:
            name, param = elem
            parent_names = name.split(".")[:-1]
            parent_module = reduce(getattr, [self] + parent_names)

            if hasattr(parent_module, 'active'):
                param.active = parent_module.active
            if hasattr(parent_module, 'layer_index'):
                param.layer_index = parent_module.layer_index
            if hasattr(parent_module, "initial_lr"):
                param.initial_lr = parent_module.initial_lr
            # TODO you have to add all attributes to "param"
            yield elem





# ---------------------------------- LOGGING
def log_lr_freezeout(layer_index, lr, iteration, writer):
    layer_key_tag = f'freezeout_layer_{layer_index}_lr'
    if writer is not None:
        writer.add_scalar(f'Learning Rate/{layer_key_tag}', lr, iteration)

def log_lr_non_freezeout(lr,iteration, writer):
    tag = 'non_freezeout_layers_lr'
    if writer is not None:
        writer.add_scalar(f'Learning Rate/{tag}', lr, iteration)





# ---------------------------------- PARAM GROUPS OPS
def create_param_groups(model: nn.Module, default_weight_decay=1e-5, default_lr=1e-3, log_writer=None):
    """Create param_groups different for freezeout layers (with attribute active), and non-freezeout layers.
    Weight decay is added to non-bias and non normalization(?) layers only.

    Args:
        model (nn.Module): _description_
        default_weight_decay (_type_, optional): _description_. Defaults to 1e-5.
        default_lr (_type_, optional): _description_. Defaults to 1e-3.

    Returns:
        list of param_group dicts: all_param_groups
    """
    layer_specific_param_groups = [] # for freezeout layers
    standard_param_groups = [] # for regular layers
    leaf_param_count = len(model.parameters())

    # NOTE named modules is not what I want to iterate over, it is recursive.
    for name, param in model.named_parameters(): # TODO modify named_modules with named_parameters() and groups parameters accordingly if it is a solution.
        # TODO answered here https://chat.openai.com/share/5d8c4fac-62ce-4ce9-970d-ecb05898b425, fix this.
        # TODO currently the param_groups added does not correspond to layer param_groups directly. But they should.
        if not param.requires_grad:
            continue
        if hasattr(param, 'active'):
            param_group = {'params': [param], 'lr': param.initial_lr, 'layer_index': param.layer_index}
            # scaling factor (gamma) and the shift (beta), which are both learnable parameters 1-D, also normalization or bias will have 0 wd
            param_group['weight_decay'] = 0. if (param.ndim <= 1 or name.endswith(".bias")) else default_weight_decay 
            layer_specific_param_groups.append(param_group)
        else:
            param_group = {'params': [param], 'lr': default_lr}
            param_group['weight_decay'] = 0. if (param.ndim <= 1 or name.endswith(".bias")) else default_weight_decay 
            standard_param_groups.append({'params': [param], 'weight_decay': 0., 'lr': default_lr})

    if log_writer:
        log_text = "Number of leaf parameter is: {}".format(leaf_param_count)
        print(log_text)
        log_writer.add_text('Info', log_text)

    # Combine the two
    all_param_groups = layer_specific_param_groups + standard_param_groups
    return all_param_groups


# NOTE timms optim_factor method used in original local mim.
def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_param_groups(optimizer, test=False, log_writer=None):
    """To access the param_groups with specific layer_indexes of the freezeout layers faster.
    NOTE that changes of the optimizer param_groups will reflect to the param_groups in freezeout_param_groups
    as they point to the same objects."""
    non_freezeout_param_groups = []
    freezeout_param_groups = {}
    for param_group in optimizer.param_groups:
        layer_index = param_group.get("layer_index")
        if layer_index is not None:
            # NOTE layer specific param_groups are added to the same list.
            if freezeout_param_groups.get(layer_index) is None:
                freezeout_param_groups[layer_index] = [param_group]
            else:
                freezeout_param_groups[layer_index].append(param_group)
        else:
            non_freezeout_param_groups.append(param_group)
    if not test: # not necessary for testing.
        print(len(list(optimizer.param_groups)))
        print(len(non_freezeout_param_groups))
        for fo_param_group in freezeout_param_groups.values():
            for param in fo_param_group:
                print(param.keys())
        assert len(freezeout_param_groups) > 15, f"freezeout_param_group_count should be at least around 15 (params), but is: {len(freezeout_param_groups)}"
        assert len(non_freezeout_param_groups) > 15, f"freezeout_param_group_count should be at least around 15 (params), but is: {len(non_freezeout_param_groups)}"
    if log_writer:
        log_text_fo = "Freezeout layer count is: {}".format(len(freezeout_param_groups))
        print(log_text_fo)
        log_writer.add_text('Info', log_text_fo)
        log_text_nfo = "Non_freezeout layer count is: {}".format(len(non_freezeout_param_groups))
        print(log_text_nfo)
        log_writer.add_text('Info', log_text_nfo)
    param_groups = {"freezeout": freezeout_param_groups,"non_freezeout": non_freezeout_param_groups}
    return param_groups

def validate_same_objects(optimizer, freezeout_param_groups):
    """Assert that changes of the optimizer param_groups will reflect to the freezeout_param_groups."""
    for param_group in optimizer.param_groups:
        if hasattr(param_group, "layer_index"):
            layer_index = param_group['layer_index']
            assert param_group in freezeout_param_groups[layer_index], "Optimizer param_group objects are not available in freezeout_param_groups"






# ---------------------------------- ADJUST LEARNING RATES
def adjust_learning_rate_freezeout(model, optimizer, epoch, cur_local_iteration, param_groups, iter_per_epoch, args, writer, test=False):
    """Freezeout decay the learning rate with half-cycle cosine after linnear warmup, step=iteration"""
    total_warmup_iterations = iter_per_epoch*args.warmup_epochs
    cur_global_iteration = cur_local_iteration + epoch*iter_per_epoch
    fractional_epoch = epoch + cur_local_iteration/iter_per_epoch # cur_global_iteration / iter_per_epoch
    if cur_global_iteration < total_warmup_iterations:
        # Update all param groups equally in warm-up iterations
        lr = args.lr*cur_global_iteration/total_warmup_iterations
        for param_group in optimizer.param_groups:
            assert "lr_scale" not in param_group, "lr_scale should be only in fine tuning"
            param_group["lr"] = lr
    else:
        lmim_cosine_lr = args.min_lr+(args.lr-args.min_lr)*0.5*(1.+math.cos(math.pi*(fractional_epoch-args.warmup_epochs)/(args.epochs-args.warmup_epochs)))
        freezeout_param_groups = param_groups["freezeout"]
        non_freezeout_param_groups = param_groups["non_freezeout"]
        update_non_freezeout_layers_lr(non_freezeout_param_groups, lmim_cosine_lr, cur_global_iteration, writer=writer)
        update_freezeout_layers_lr(model, cur_global_iteration, optimizer, freezeout_param_groups, writer=writer, test=test)
        validate_same_objects(optimizer, freezeout_param_groups)

def update_freezeout_layers_lr(model, cur_global_iteration, optim, freezeout_param_groups, writer, test=False):
        """initial_lr: The default learning rate of the overall model before scaling (after warmup)
        Here we assume the min_lr=0 in cosine annealing (orginally -> min_lr + (lr-min_lr)*...)"""
        # NOTE cur_global_iteration incremented by train loop
        # Loop over all modules, requires -> cur_global_iteration and module. active, max_iteration, layer_index,
        freezeout_active_layer_count = 0
        for m in model.modules(): # TODO this will work if all active layers are captured by this if statement.
            # If a module is active and at the freezeout layer level of model.modules() hierarchy.:
            if hasattr(m,'freezeout_module_level_specifier') and m.active: # NOTE does not enter if no more active. TODO check if fixed problem.
                freezeout_active_layer_count += 1
                target_freezeout_param_group = freezeout_param_groups[m.layer_index]
                # If we've passed this layer's freezing point, deactivate it.
                if cur_global_iteration > m.max_iteration: 
                    lr = 0
                    m.active = False
                    m.requires_grad = False # NOTE detach is no longer necessary in the forward passes.
                    # Also make sure we remove all this layer from the optimizer
                    # optim.param_groups.remove(target_freezeout_param_group) -> default one.
                    for target_freezeout_param in target_freezeout_param_group:
                        optim.param_groups.remove(target_freezeout_param)
                        if target_freezeout_param is not None:
                            del target_freezeout_param # NOTE assumes that this parameter will not be activated (trainable) again
                    del freezeout_param_groups[m.layer_index] # NOTE delete the obsolete param groups.
                else:
                    # update the LR
                    layer_wise_initial_lr = m.initial_lr # NOTE lr_ratio already scaled lrs per layer
                    lr = (layer_wise_initial_lr/2)*(1+np.cos(np.pi*cur_global_iteration/m.max_iteration))
                    for target_freezeout_param in target_freezeout_param_group:
                        target_freezeout_param['lr'] = lr
                # Add the learning rate of this layer to the log
                log_lr_freezeout(layer_index=m.layer_index, lr=lr, iteration=cur_global_iteration, writer=writer)
        assert freezeout_active_layer_count == len(freezeout_param_groups), "optimizer's freezeout_param_groups should all be updated"
        if cur_global_iteration < 50: # assert only for initial iterations
            if not test: # Do not assert for testing.
                assert freezeout_active_layer_count > 15, "freezeout_active_layer_count should be at least around 20 (layers)"


def update_non_freezeout_layers_lr(non_freezeout_param_groups, lmim_cosine_lr, cur_global_iteration, writer):
    """This method updates non-freezeout layers lr.
    Cosine annealng applied previously to lr is lmim_cosine_lr."""
    for non_freezeout_param_group in non_freezeout_param_groups:
        non_freezeout_param_group['lr'] = lmim_cosine_lr
    log_lr_non_freezeout(lr=lmim_cosine_lr, iteration=cur_global_iteration, writer=writer)