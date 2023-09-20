import numpy as np
from timm.models.vision_transformer import Block # NOTE has internal skip connections.
from torch import nn
import math


# NOTE checked oK.
def create_param_groups(model: nn.Module, default_weight_decay=1e-5, default_lr=1e-3):
    """Create param_groups different for freezeout layers (with attribute active), and non-freezeout layers.
    Weight decay is added to non-bias and non normalization(?) layers only.

    Args:
        model (nn.Module): _description_
        default_weight_decay (_type_, optional): _description_. Defaults to 1e-5.
        default_lr (_type_, optional): _description_. Defaults to 1e-3.

    Returns:
        list of param_group dicts: all_param_groups
    """
    no_decay = []
    decay = []
    layer_specific_param_groups = []

    for name, module in model.named_modules():
        if hasattr(module, 'active'):
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                # NOTE this part creates a new parameter group with or without weight decay.
                # Add the lr and layer_index attributes, they must exist in the module
                param_group = {'params': param, 'lr': module.lr, 'layer_index': module.layer_index}

                # scaling factor (gamma) and the shift (beta), which are both learnable parameters 1-D
                if param.ndim <= 1 or name.endswith(".bias"): # normalization or bias
                    param_group['weight_decay'] = 0.
                else:
                    param_group['weight_decay'] = default_weight_decay

                layer_specific_param_groups.append(param_group)

        else:
            for param in module.parameters():
                if not param.requires_grad:
                    continue

                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

    # Create regular decay and no_decay groups, you need to specify lr here because it is not given by the optimizer.
    standard_param_groups = [
        {'params': no_decay, 'weight_decay': 0., 'lr': default_lr},
        {'params': decay, 'weight_decay': default_weight_decay, 'lr': default_lr}
    ]

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


def adjust_learning_rate_fo(model, optimizer, epoch, cur_local_iteration, param_groups, iter_per_epoch, args):
    """Freezeout decay the learning rate with half-cycle cosine after linnear warmup, step=iteration"""
    total_warmup_iterations = iter_per_epoch*args.warmup_epochs
    cur_global_iteration = cur_local_iteration + epoch*iter_per_epoch
    if cur_global_iteration < total_warmup_iterations:
        # Update all param groups equally in warm-up iterations
        lr = args.lr*cur_global_iteration/total_warmup_iterations
        for param_group in optimizer.param_groups:
            assert "lr_scale" not in param_group, "lr_scale should be only in fine tuning"
            param_group["lr"] = lr
    else:
        lmim_cosine_lr = args.min_lr+(args.lr-args.min_lr)*0.5*(1.+math.cos(math.pi*(epoch-args.warmup_epochs)/(args.epochs-args.warmup_epochs)))
        freezeout_param_groups = param_groups["freezeout"]
        non_freezeout_param_groups = param_groups["non_freezeout"]
        update_non_freezeout_layers_lr(non_freezeout_param_groups, lmim_cosine_lr)
        update_freezeout_layers_lr(model, cur_global_iteration, optimizer, freezeout_param_groups, initial_lr=args.lr)
        validate_same_objects(optimizer, freezeout_param_groups) #

def update_freezeout_layers_lr(model, cur_global_iteration, optim, freezeout_param_groups, initial_lr):
        """initial_lr: The default learning rate of the overall model before scaling (after warmup)
        Here we assume the min_lr=0 in cosine annealing (orginally -> min_lr + (lr-min_lr)*...)"""
        # NOTE cur_global_iteration incremented by train loop
        # Loop over all modules, requires -> cur_global_iteration and module. active, max_iteration, layer_index,
        active_attr_count = 0
        for m in model.modules():
            # If a module is active:
            active_attr_count = active_attr_count + 1 if hasattr(m,'active') else active_attr_count
            if hasattr(m,'active') and m.active:
                target_freezeout_param_group = freezeout_param_groups[m.layer_index]
                # If we've passed this layer's freezing point, deactivate it.
                if cur_global_iteration > m.max_iteration: 
                    m.active = False
                    m.requires_grad = False # NOTE detach is no longer necessary in the forward passes.
                    # Also make sure we remove all this layer from the optimizer
                    del freezeout_param_groups[m.layer_index] # NOTE UTKU is this line necessary?
                    optim.param_groups.remove(target_freezeout_param_group)
                else:
                    # update the LR
                    layer_wise_initial_lr = m.initial_lr # NOTE lr_ratio scaled lrs per layer
                    lr = (layer_wise_initial_lr/2)*(1+np.cos(np.pi*cur_global_iteration/m.max_iteration))
                    target_freezeout_param_group['lr'] = lr
        assert active_attr_count > 15, "active_attr_count should be at least around 20 (layers)"

def update_non_freezeout_layers_lr(non_freezeout_param_groups, lmim_cosine_lr):
    for non_freezeout_param_group in non_freezeout_param_groups:
        non_freezeout_param_group['lr'] = lmim_cosine_lr

def get_param_groups(optimizer):
    """To access the param_groups with specific layer_indexes of the freezeout layers faster.
    NOTE that changes of the optimizer param_groups will reflect to the param_groups in freezeout_param_groups
    as they point to the same objects."""
    non_freezeout_param_groups = {}
    freezeout_param_groups = {}
    for param_group in optimizer.param_groups:
        if hasattr(param_group, "layer_index"):
            layer_index = param_group['layer_index']
            freezeout_param_groups[layer_index] = param_group
        else:
            non_freezeout_param_groups.append(param_group)
    assert len(freezeout_param_groups) > 15, "freezeout_param_group_count should be at least around 20 (layers)"
    assert len(non_freezeout_param_groups) > 5, "freezeout_param_group_count should be at least around 5 (layers)"
    print("Freezeout layer count is: ", len(freezeout_param_groups))
    print("Non_freezeout layer count is: ", len(non_freezeout_param_groups))
    param_groups = {"freezeout": freezeout_param_groups,"non_freezeout": non_freezeout_param_groups}
    return param_groups

def validate_same_objects(optimizer, freezeout_param_groups):
    """Assert that changes of the optimizer param_groups will reflect to the freezeout_param_groups."""
    for param_group in optimizer.param_groups:
        if hasattr(param_group, "layer_index"):
            layer_index = param_group['layer_index']
            assert param_group is freezeout_param_groups[layer_index], "Objects are not the same"
