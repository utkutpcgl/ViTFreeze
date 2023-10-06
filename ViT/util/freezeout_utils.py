import numpy as np
from timm.models.vision_transformer import Block # NOTE has internal skip connections.
from torch import nn
import math
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/lr_logging_experiment")

LR_LOG_FO = {'iteration': []}
LR_LOG_NFO = {'iteration': [], "non_freezeout_layers_lr": []}
# LOG_ITERATION_FREQ = 5 # log per iteration frequency
# PLOT_LR_FREQ = 5

# LR VIS AND LOGGING
# def log_and_plot_lrs(cur_global_iteration):
#     # if cur_global_iteration % LOG_ITERATION_FREQ == 0:
#     save_xl(cur_global_iteration) # log to excel
#     # if cur_global_iteration % PLOT_LR_FREQ == 0:
#     plot_learning_rates()

def log_lr_fo(layer_index, lr, iteration):
    layer_key_tag = f'freezeout_layer_{layer_index}_lr'
    writer.add_scalar(f'Learning Rate/{layer_key_tag}', lr, iteration)

def log_lr_nfo(lr,iteration):
    tag = 'non_freezeout_layers_lr'
    writer.add_scalar(f'Learning Rate/{tag}', lr, iteration)

# def save_xl(cur_global_iteration):
#     # save_xl is not called simulatenously with lr logging, this can cause iteration lr mismatch fix it.
#     # Log the current iteration
#     LR_LOG_FO['iteration'].append(cur_global_iteration)
#     LR_LOG_NFO['iteration'].append(cur_global_iteration)
#     # Save learning rates to Excel
#     pd.DataFrame(LR_LOG_FO).to_excel('learning_rates_fo.xlsx', index=False)
#     pd.DataFrame(LR_LOG_NFO).to_excel('learning_rates_nfo.xlsx', index=False)

# def plot_learning_rates():
#     df_fo = pd.read_excel('learning_rates_fo.xlsx')
#     for column in df_fo.columns:
#         if column != 'iteration':
#             plt.plot(df_fo['iteration'], df_fo[column], label=column)
#     plt.legend()
#     plt.xlabel('Iteration')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rates per Layer per Iteration FO')
#     plt.savefig('learning_rates_fo.png')

#     df_nfo = pd.read_excel('learning_rates_nfo.xlsx')
#     for column in df_nfo.columns:
#         if column != 'iteration':
#             plt.plot(df_nfo['iteration'], df_nfo[column], label=column)
#     plt.legend()
#     plt.xlabel('Iteration')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rates per Layer per Iteration NFO')
#     plt.savefig('learning_rates_nfo.png')
    


# LR UPDATING
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
    layer_specific_param_groups = [] # for freezeout layers
    standard_param_groups = [] # for regular layers
    layer_count = 0

    # NOTE named modules is not what I want to iterate over, it is recursive.
    for name, module in model.named_modules():
        if not list(module.children()):  # if module has no child modules, it's a leaf
            layer_count += 1
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
                        continue # skip param if already in processed params

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

    # Create regular decay and no_decay groups, you need to specify lr here because it is not given by the optimizer.
    if len(no_decay) != 0:
        standard_param_groups.append({'params': no_decay, 'weight_decay': 0., 'lr': default_lr})
    if len(decay) != 0:
        standard_param_groups.append({'params': decay, 'weight_decay': default_weight_decay, 'lr': default_lr})

    print("Number of layers is:", layer_count)

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


def adjust_learning_rate_fo(model, optimizer, epoch, cur_local_iteration, param_groups, iter_per_epoch, args, test=False):
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
        update_non_freezeout_layers_lr(non_freezeout_param_groups, lmim_cosine_lr, cur_global_iteration)
        update_freezeout_layers_lr(model, cur_global_iteration, optimizer, freezeout_param_groups, initial_lr=args.lr, test=test)
        validate_same_objects(optimizer, freezeout_param_groups)
        # log_and_plot_lrs(cur_global_iteration)


def update_freezeout_layers_lr(model, cur_global_iteration, optim, freezeout_param_groups, initial_lr, test=False):
        """initial_lr: The default learning rate of the overall model before scaling (after warmup)
        Here we assume the min_lr=0 in cosine annealing (orginally -> min_lr + (lr-min_lr)*...)"""
        # NOTE cur_global_iteration incremented by train loop
        # Loop over all modules, requires -> cur_global_iteration and module. active, max_iteration, layer_index,
        active_attr_count = 0
        for m in model.modules():
            # If a module is active:
            if hasattr(m,'active') and m.active:
                active_attr_count += 1
                # TODO adjust this part for listed target_freezeout_param_groups being a list
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
                    # Prev method:
                    # if target_freezeout_param_group is not None: # delete if exists
                    #     del freezeout_param_groups[m.layer_index] # NOTE UTKU is this line necessary?
                else:
                    # update the LR
                    layer_wise_initial_lr = m.initial_lr # NOTE lr_ratio scaled lrs per layer
                    lr = (layer_wise_initial_lr/2)*(1+np.cos(np.pi*cur_global_iteration/m.max_iteration))
                    # target_freezeout_param_group['lr'] = lr -> prev line.
                    for target_freezeout_param in target_freezeout_param_group:
                        target_freezeout_param['lr'] = lr
                # Add the learning rate of this layer to the log
                log_lr_fo(layer_index=m.layer_index, lr=lr, iteration=cur_global_iteration)
        if cur_global_iteration < 50: # assert only for initial iterations
            if not test: # Do not assert for testing.
                assert active_attr_count > 15, "active_attr_count should be at least around 20 (layers)"


def update_non_freezeout_layers_lr(non_freezeout_param_groups, lmim_cosine_lr, cur_global_iteration):
    """This method updates non-freezeout layers lr.
    Cosine annealng applied previously to lr is lmim_cosine_lr."""
    for non_freezeout_param_group in non_freezeout_param_groups:
        non_freezeout_param_group['lr'] = lmim_cosine_lr
    log_lr_nfo(lr=lmim_cosine_lr, iteration=cur_global_iteration)


def get_param_groups(optimizer, test=False):
    """To access the param_groups with specific layer_indexes of the freezeout layers faster.
    NOTE that changes of the optimizer param_groups will reflect to the param_groups in freezeout_param_groups
    as they point to the same objects."""
    non_freezeout_param_groups = []
    freezeout_param_groups = {}
    for param_group in optimizer.param_groups:
        print(param_group.get("layer_index"))
        layer_index = param_group.get("layer_index")
        if layer_index is not None:
            # NOTE freezeout params are added to a list
            if freezeout_param_groups.get(layer_index) is None:
                freezeout_param_groups[layer_index] = [param_group]
            else:
                freezeout_param_groups[layer_index].append(param_group)
        else:
            non_freezeout_param_groups.append(param_group)
    if not test: # not necessary for testing.
        assert len(freezeout_param_groups) > 15, "freezeout_param_group_count should be at least around 20 (layers)"
        assert len(non_freezeout_param_groups) > 5, "freezeout_param_group_count should be at least around 5 (layers)"
    print("Freezeout layer count is: ", len(freezeout_param_groups))
    print(freezeout_param_groups)
    print("Non_freezeout layer count is: ", len(non_freezeout_param_groups))
    print(non_freezeout_param_groups)
    param_groups = {"freezeout": freezeout_param_groups,"non_freezeout": non_freezeout_param_groups}
    return param_groups

def validate_same_objects(optimizer, freezeout_param_groups):
    """Assert that changes of the optimizer param_groups will reflect to the freezeout_param_groups."""
    for param_group in optimizer.param_groups:
        if hasattr(param_group, "layer_index"):
            layer_index = param_group['layer_index']
            assert param_group is freezeout_param_groups[layer_index], "Objects are not the same"







# DUMMY MODEL TO VALIDATE LR UPDATE FUNCTIONALITY
class DummyModel(nn.Module):
    # Dummy model with `active` attributes and layer indices
    def __init__(self):
        super(DummyModel, self).__init__()
        # self.layer1 = Block(dim=32, num_heads=2, mlp_ratio=1.0)
        self.layer1 = nn.Linear(10, 10)
        self.layer1.lr = 0.001
        self.layer1.active = True
        self.layer1.layer_index = 0
        self.layer1.max_iteration = 100
        self.layer1.initial_lr = 0.01
        self.layer2 = nn.Linear(10, 10)
        self.layer2.lr = 0.001
        self.layer2.lr = 0.001
        self.layer2.active = True
        self.layer2.layer_index = 1
        self.layer2.max_iteration = 100
        self.layer2.initial_lr = 0.01


# Dummy Args
class Args:
    def __init__(self):
        self.lr = 0.01
        self.min_lr = 0.001
        self.warmup_epochs = 1
        self.epochs = 5

# Simulation of lr logging and plotting
def simulate_lr_logging():
    import torch.optim as optim
    model = DummyModel()
    args = Args()

    # Using the provided function to create parameter groups
    param_groups = create_param_groups(model)
    optimizer = optim.SGD(param_groups, lr=args.lr)

    param_groups = get_param_groups(optimizer, test=True)
    iter_per_epoch = 10

    for epoch in range(args.epochs):
        for cur_local_iteration in range(iter_per_epoch):
            adjust_learning_rate_fo(model, optimizer, epoch, cur_local_iteration, param_groups, iter_per_epoch, args, test=True)
    writer.close()

# Run simulation
if __name__ == "__main__":
    simulate_lr_logging()
