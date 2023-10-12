from torch import nn
from freezeout_utils import create_param_groups, get_param_groups, adjust_learning_rate_freezeout
from torch.utils.tensorboard import SummaryWriter

test_writer = SummaryWriter("runs/lr_logging_experiment")

scale_fn = {'linear':lambda x: x,
            'squared': lambda x: x**2,
            'cubic': lambda x: x**3}

# DUMMY MODEL TO VALIDATE LR UPDATE FUNCTIONALITY
class DummyModel(nn.Module):
    # Dummy model with `active` attributes and layer indices
    def __init__(self):
        super(DummyModel, self).__init__()
        # NOTE increase number of layers of this model with the same logic (30 layers), with minimal number of reasonable lines of code.
        # self.layer1 = Block(dim=32, num_heads=2, mlp_ratio=1.0)
        # Creating 30 layers with the specified logic
        for i in range(1, 31):  
            setattr(self, f'layer{i}', nn.Linear(10, 10))
            layer = getattr(self, f'layer{i}')
            layer.active = True
            layer.layer_index = i - 1  # 0-based index

        self.how_scale = "cubic"
        self.t_0 = 0.5
        self.layer_index = 30  # Updated total number of layers
        self.scale_lr = True


# Dummy Args
class Args:
    def __init__(self):
        self.lr = 0.001 # default lr
        self.min_lr = 0.0001
        self.warmup_epochs = 0
        self.epochs = 10

# Simulation of lr logging and plotting
def simulate_lr_logging(log_writer):
    import torch.optim as optim
    model = DummyModel()

    args = Args()

    iter_per_epoch = 10
    # Freezeout lr setting logic
    lr_scale_fn = scale_fn[model.how_scale] # freezeout spec
    t_0 = model.t_0 # freezeout spec
    num_of_layers = model.layer_index # freezeout spec
    for module in model.modules():
        if hasattr(module,'active'): # freezout specific
            # the ratio to be multiplied with the initial learning rate.
            module.lr_ratio = lr_scale_fn(t_0 + (1 - t_0) * float(module.layer_index) / num_of_layers) # freezout specific
            module.initial_lr = args.lr/module.lr_ratio if model.scale_lr else args.lr # freezout specific
            # NOTE iterations set auto instead of 1000 (so in freezeout), warmup is not included.
            module.max_iteration = (args.epochs-args.warmup_epochs) * iter_per_epoch * module.lr_ratio
            # TODO Log lr_ratio and max iteration per layer to log_writer.
            log_writer.add_scalar('LR Ratios', module.lr_ratio, module.layer_index)
            log_writer.add_scalar('Max Iterations', module.max_iteration, module.layer_index)

    # Using the provided function to create parameter groups
    param_groups = create_param_groups(model, log_writer=log_writer)
    optimizer = optim.SGD(param_groups)

    param_groups = get_param_groups(optimizer, test=True,log_writer=log_writer)

   


    for epoch in range(args.epochs):
        for cur_local_iteration in range(iter_per_epoch):
            adjust_learning_rate_freezeout(model, optimizer, epoch, cur_local_iteration, param_groups, iter_per_epoch, args, writer=log_writer, test=True)

# Run simulation
if __name__ == "__main__":
    simulate_lr_logging(log_writer=test_writer)