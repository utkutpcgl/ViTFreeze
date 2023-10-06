from torch import nn
from freezeout_utils import create_param_groups, get_param_groups, adjust_learning_rate_freezeout
from torch.utils.tensorboard import SummaryWriter
test_writer = SummaryWriter("runs/lr_logging_experiment")


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
def simulate_lr_logging(log_writer):
    import torch.optim as optim
    model = DummyModel()
    args = Args()

    # Using the provided function to create parameter groups
    param_groups = create_param_groups(model, log_writer=log_writer)
    optimizer = optim.SGD(param_groups, lr=args.lr)

    param_groups = get_param_groups(optimizer, test=True,log_writer=log_writer)
    iter_per_epoch = 10

    for epoch in range(args.epochs):
        for cur_local_iteration in range(iter_per_epoch):
            adjust_learning_rate_freezeout(model, optimizer, epoch, cur_local_iteration, param_groups, iter_per_epoch, args, test=True)

# Run simulation
if __name__ == "__main__":
    simulate_lr_logging(log_writer=test_writer)