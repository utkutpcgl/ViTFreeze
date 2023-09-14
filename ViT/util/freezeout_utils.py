import numpy as np
from timm.models.vision_transformer import Block # NOTE has internal skip connections.
from torch import nn


# https://chat.openai.com/share/30777d71-4944-41ad-80a3-17dfca5bac7a
# TODO  is there a need for this block to detach and have freezeout attributes?? No I guess.
# class CustomBlock(Block):
#     def __init__(self, *args, layer_index=0, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         # Add your custom variables
#         self.active = True
#         self.layer_index = layer_index

#     def forward(self, x):
#         if self.active:
#             return super().forward(x)
#         else:
#             # Custom behavior when the block is inactive
#             # For example, you might want to detach the tensor to stop gradients
#             return x.detach()


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


def adjust_learning_rate_fo(model, optimizer, epoch, cur_local_iteration, iter_per_epoch, args):
    # TODO Double check here
    """Freezeout decay the learning rate with half-cycle cosine after linnear warmup, step=iteration"""
    total_warmup_iterations = iter_per_epoch*args.warmup_epochs
    cur_global_iteration = cur_local_iteration + epoch*iter_per_epoch
    if cur_global_iteration < total_warmup_iterations:
        lr = args.lr*cur_global_iteration/total_warmup_iterations
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group: # NOTE UTKU Only used for fine tuning.
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
    else:
        # NOTE not used anywhere, but initial lr after warmup is (Optionally scale the learning rates to have the same total):
        # module.lr = args.lr / module.lr_ratio if scale_lr else args.lr # freezout specific,
        update_lr(model, cur_global_iteration, optimizer, initial_lr=args.lr)

def update_lr(model, cur_global_iteration, optim, initial_lr):
        """
        initial_lr: The default learning rate of the overall model before scaling (after warmup)
        Here we assume the min_lr=0 in cosine annealing (orginally -> min_lr + (lr-min_lr)*...)
        """
        # NOTE cur_global_iteration incremented by train loop
        # Loop over all modules, requires -> cur_global_iteration and module. active, max_iteration, layer_index, lr_ratio,
        for m in model.modules():
            # If a module is active:
            if hasattr(m,'active') and m.active:
                # If we've passed this layer's freezing point, deactivate it.
                if cur_global_iteration > m.max_iteration: 
                    m.active = False
                    m.requires_grad = False # NOTE detach is no longer necessary in the forward passes.
                    # Also make sure we remove all this layer from the optimizer
                    for i,group in enumerate(optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            optim.param_groups.remove(group)
                # If not, update the LR
                else:
                    for i,group in enumerate(optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            lr = initial_lr/m.lr_ratio if model.scale_lr else initial_lr
                            optim.param_groups[i]['lr'] = (lr/2)*(1+np.cos(np.pi*cur_global_iteration/m.max_iteration))