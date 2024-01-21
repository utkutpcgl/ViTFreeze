import numpy as np
from timm.models.vision_transformer import Block # NOTE has internal skip connections.
from torch import nn
import math
import pandas as pd
from functools import reduce
import torch
from multiprocessing import Pool, cpu_count
from numba import jit

DECODER_EXTRA_NORM_LAYER_COUNT = 4 # NOTE normalization layers fed to decoder adds 4 more neihbouring layers (len(model.ID))
DECODER_EXTRA_LAYER_COUNT = 4 # NOTE decoder adds 4 more neihbouring layers (len(model.ID))
HOG_EXTRA_LAYER_COUNT = 4 # NOTE HOG adds 4 more neihbouring layers
FREEZEOUT_LAYER_COUNT_VIT_B = 13 + DECODER_EXTRA_NORM_LAYER_COUNT + DECODER_EXTRA_LAYER_COUNT + HOG_EXTRA_LAYER_COUNT # There are 13 freezable blocks (layers) in the transformer.
ITERATION_LOG_PERIOD = 625 # log every # iterations


# ---------------------------------- ATTRIBUTE AWARE LEAVES
class AttributeAwareModule(nn.Module):
    def named_parameters(self, prefix='', recurse=True):
        gen = super().named_parameters(prefix=prefix, recurse=recurse)
        for name, param in gen: # TODO there is a mistake, this sets all parents with active attr. SOLVED?
            parent_names = name.split('.')[:-1]
            for i in range(len(parent_names), 0, -1):
                sub_parent_names = parent_names[:i]
                parent_module = reduce(getattr, [self] + sub_parent_names)
                # Add attributes from parent modules
                if hasattr(parent_module, 'active'):
                    setattr(param, 'active', getattr(parent_module, 'active'))
                if hasattr(parent_module, 'layer_index'):
                    setattr(param, 'layer_index', getattr(parent_module, 'layer_index'))
                if hasattr(parent_module, 'initial_lr'):
                    setattr(param, 'initial_lr', getattr(parent_module, 'initial_lr'))
            yield name, param





# ---------------------------------- LOGGING
def log_lr_freezeout(layer_index, lr, iteration, writer):
    if iteration % ITERATION_LOG_PERIOD != 0:
        return
    layer_key_tag = f'freezeout_layer_{layer_index}_lr'
    if writer is not None:
        writer.add_scalar(f'Learning Rate/{layer_key_tag}', lr, iteration)

def log_lr_non_freezeout(lr, iteration, writer):
    if iteration % ITERATION_LOG_PERIOD != 0:
        return
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
    leaf_param_count = len(list(model.parameters()))

    # NOTE named modules is not what I want to iterate over, it is recursive.
    for name, param in model.named_parameters():
        # NOTE Leaf parameters are separated to 4 groups (fo-nfo, wd-nwd)
        if not param.requires_grad:
            continue
        if hasattr(param, 'active'):
            param_group = {'params': [param], 'lr': param.initial_lr, 'layer_index': param.layer_index} # NOTE no need for activeness information
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
        if len(freezeout_param_groups) >= 13:
            print(f"number of freezeout layers should be at least around 13, but is: {len(freezeout_param_groups)}")
        if len(non_freezeout_param_groups) >= 85:
            print(f"freezeout_param_group_count should be at least around 15 (params), but is: {len(non_freezeout_param_groups)}")
    if log_writer:
        log_text_fo = "Freezeout layer count is: {}".format(len(freezeout_param_groups))
        print(log_text_fo)
        log_writer.add_text('Info', log_text_fo)
        log_text_nfo = "Non_freezeout param count is: {}".format(len(non_freezeout_param_groups))
        print(log_text_nfo)
        log_writer.add_text('Info', log_text_nfo)
    param_groups = {"freezeout": freezeout_param_groups,"non_freezeout": non_freezeout_param_groups}
    return param_groups





# PARAM GROUP COMPARISON
def are_tensors_equal(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        return False
    return torch.all(torch.eq(tensor1, tensor2)).item()

def are_dicts_equal(dict1, dict2):
    """Default dictionary comparison fails due to tensor equality comparison."""
    if dict1.keys() != dict2.keys():
        return False
    
    for k, v in dict1.items():
        if isinstance(v, list): # The element inside the list is a tensor
            if isinstance(v[0], torch.Tensor):
                if not are_tensors_equal(v[0], dict2[k][0]):
                    return False
        elif dict1[k] != dict2[k]:
            return False
    return True

def validate_same_objects(optimizer, freezeout_param_groups):
    """Assert that changes of the optimizer param_groups will reflect to the freezeout_param_groups."""
    for param_group in optimizer.param_groups:
        if 'layer_index' in param_group:
            layer_index = param_group['layer_index']
            # TODO this returned false.
            assert any(are_dicts_equal(param_group, other_group) for other_group in freezeout_param_groups[layer_index]), "Optimizer param_group objects are not available in freezeout_param_groups"
    #TODO there is mistake in param_group element comparision.
    for param_group_list in freezeout_param_groups.values():
        for param_group in param_group_list:
            assert any(are_dicts_equal(param_group, other_group) for other_group in optimizer.param_groups), "freezeout_param_groups param_group objects are not available in optimizer param_groups"




# ---------------------------------- ADJUST LEARNING RATES
def adjust_learning_rate_freezeout(optimizer, epoch, cur_local_iteration, param_groups, active_freezeout_modules, iter_per_epoch, args, writer, non_layerwise_lr):
    """Freezeout decay the learning rate with half-cycle cosine after linnear warmup, step=iteration"""
    total_warmup_iterations = iter_per_epoch*args.warmup_epochs
    cur_global_iteration = cur_local_iteration + epoch*iter_per_epoch
    cur_global_iteration_warmup_subtracted = cur_global_iteration - total_warmup_iterations
    fractional_epoch = epoch + cur_local_iteration/iter_per_epoch # cur_global_iteration / iter_per_epoch
    if cur_global_iteration < total_warmup_iterations:
        min_active_layer_index = 0
        # Update all param groups equally in warm-up iterations
        lr = args.lr*cur_global_iteration/total_warmup_iterations
        for param_group in optimizer.param_groups:
            assert "lr_scale" not in param_group, "lr_scale should be only in fine tuning"
            param_group["lr"] = lr
    else:
        regular_cosine_lr = args.min_lr+(args.lr-args.min_lr)*0.5*(1.+math.cos(math.pi*(fractional_epoch-args.warmup_epochs)/(args.epochs-args.warmup_epochs))) 
        # TODO the param_groups might not modify all necessary param groups of the optimizer if t0=1 training does not match original pre-training results (with not_scale_lr and non_laywerise_lr set True)
        freezeout_param_groups = param_groups["freezeout"]
        non_freezeout_param_groups = param_groups["non_freezeout"]
        update_non_freezeout_layers_lr(non_freezeout_param_groups, regular_cosine_lr, cur_global_iteration, writer=writer)
        min_active_layer_index = update_freezeout_layers_lr(cur_global_iteration, cur_global_iteration_warmup_subtracted, optimizer, freezeout_param_groups, active_freezeout_modules, writer=writer, non_layerwise_lr=non_layerwise_lr, regular_cosine_lr=regular_cosine_lr)
    return min_active_layer_index
        

def update_freezeout_layers_lr(cur_global_iteration, cur_global_iteration_warmup_subtracted, optim, freezeout_param_groups, active_freezeout_modules, writer, non_layerwise_lr, regular_cosine_lr):
    """initial_lr: The default learning rate of the overall model before scaling (after warmup)
    Here we assume the min_lr=0 in cosine annealing (orginally -> min_lr + (lr-min_lr)*...)"""
    # NOTE cur_global_iteration incremented by train loop
    # Loop over all modules, requires -> cur_global_iteration and module. active, max_iteration_warmup_subtracted, layer_index,
    freezeout_active_layer_set = set()
    for m in active_freezeout_modules:
        # If a module is active and at the freezeout layer level of model.modules() hierarchy.:
        if not hasattr(m,'freezeout_module_level_specifier') or not m.active:
            continue # NOTE does not enter if no more active.
        # If we've passed this layer's freezing point, deactivate it.
        target_freezeout_param_group = freezeout_param_groups.get(m.layer_index)
        if cur_global_iteration_warmup_subtracted > m.max_iteration_warmup_subtracted: 
            lr = 0
            m.active = False
            m.eval() # NOTE did not make a huge difference.
            # Also make sure we remove all this layer from the optimizer
            # optim.param_groups.remove(target_freezeout_param_group) -> default one.
            if target_freezeout_param_group is None:
                continue
            for pg_index, pg in reversed(list(enumerate(optim.param_groups))):
                if pg.get('layer_index') == m.layer_index:  # Assuming you have 'layer_index' in param_groups
                    remove_param_from_optimizer_and_grad_comp(optim,pg_index)
            del freezeout_param_groups[m.layer_index]
        else:
            freezeout_active_layer_set.add(m.layer_index) # NOTE will see same layer_index twice for decoder input layers
            # update the LR
            if non_layerwise_lr: #Dont apply layerwise lr if this is specified.
                lr = regular_cosine_lr
            else:
                layer_wise_initial_lr = m.initial_lr # NOTE lr_ratio already scaled lrs per layer
                lr = compute_lr(layer_wise_initial_lr, cur_global_iteration_warmup_subtracted, max_iteration_warmup_subtracted=m.max_iteration_warmup_subtracted)
            for target_freezeout_param in target_freezeout_param_group:
                target_freezeout_param['lr'] = lr
        # Add the learning rate of this layer to the log
        log_lr_freezeout(layer_index=m.layer_index, lr=lr, iteration=cur_global_iteration, writer=writer)
    min_active_layer_index = min(freezeout_active_layer_set)
    assert len(freezeout_active_layer_set) == len(freezeout_param_groups), "optimizer's freezeout_param_groups should all be updated"
    return min_active_layer_index

@jit(nopython=True)
def compute_lr(layer_wise_initial_lr, cur_global_iteration_warmup_subtracted, max_iteration_warmup_subtracted):
    return (layer_wise_initial_lr/2) * (1+np.cos(np.pi*cur_global_iteration_warmup_subtracted/max_iteration_warmup_subtracted))

def get_freezeout_modules(model):
    return [m for m in model.modules() if hasattr(m, 'freezeout_module_level_specifier') and m.active]


def remove_param_from_optimizer_and_grad_comp(optim, pg_index):
    # Remove corresponding state
    for param in optim.param_groups[pg_index]['params']:
        param.requires_grad = False # changed iteration speed 1/16 approx.
        if param in optim.state:
            del optim.state[param]
    del optim.param_groups[pg_index]

# ALIGN OBJECTS TO STATE_DICTS
def align_optimizer_to_checkpoint(optimizer, checkpoint_state_dict, model):
    checkpoint_layer_indexes = {pg.get('layer_index', None) for pg in checkpoint_state_dict['param_groups']}
    for m in model.modules():
        if not hasattr(m, 'freezeout_module_level_specifier') or not m.active:
            continue
        layer_index = m.layer_index
        if layer_index not in checkpoint_layer_indexes:
            for pg_index, pg in reversed(list(enumerate(optimizer.param_groups))):
                if pg.get('layer_index') == layer_index:  # Assuming you have 'layer_index' in param_groups
                    remove_param_from_optimizer_and_grad_comp(optimizer, pg_index)


def update_non_freezeout_layers_lr(non_freezeout_param_groups, regular_cosine_lr, cur_global_iteration, writer):
    """This method updates non-freezeout layers lr.
    Cosine annealng applied previously to lr is regular_cosine_lr."""
    for non_freezeout_param_group in non_freezeout_param_groups:
        non_freezeout_param_group['lr'] = regular_cosine_lr
    log_lr_non_freezeout(lr=regular_cosine_lr, iteration=cur_global_iteration, writer=writer)







