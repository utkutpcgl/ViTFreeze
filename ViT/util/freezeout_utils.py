import numpy as np
from timm.models.vision_transformer import Block # NOTE has internal skip connections.


# https://chat.openai.com/share/30777d71-4944-41ad-80a3-17dfca5bac7a
class CustomBlock(Block):
    def __init__(self, *args, layer_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add your custom variables
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if self.active:
            return super().forward(x)
        else:
            # Custom behavior when the block is inactive
            # For example, you might want to detach the tensor to stop gradients
            return x.detach()



def update_lr(mae_vit_model, optim, initial_lr):
        # Loop over all modules, requires -> self.j and module. active, max_j, layer_index, lr_ratio,
        for m in mae_vit_model.modules():
            # If a module is active:
            if hasattr(m,'active') and m.active:
                # If we've passed this layer's freezing point, deactivate it.
                if mae_vit_model.j > m.max_j: 
                    m.active = False
                    # Also make sure we remove all this layer from the optimizer
                    for i,group in enumerate(optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            optim.param_groups.remove(group)
                # If not, update the LR
                else:
                    for i,group in enumerate(optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            optim.param_groups[i]['lr'] = ((initial_lr/2)/m.lr_ratio)*(1+np.cos(np.pi*mae_vit_model.j/m.max_j))\
                                                              if mae_vit_model.scale_lr else (initial_lr/2) * (1+np.cos(np.pi*mae_vit_model.j/m.max_j))
        mae_vit_model.j += 1  # TODO left here.

