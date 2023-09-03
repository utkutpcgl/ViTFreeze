import numpy as np

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