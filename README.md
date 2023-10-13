# freezeout_localmim_rho

## Running Experiments

### Pre-training:

  Explained in the repos README.md as this, to pre-train ViT-B:
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --batch_size 256 --model MIM_vit_base_patch16 --hog_nbins 9 --mask_ratio 0.75 --epochs 1600 --warmup_epochs 40 --blr 2e-4 --weight_decay 0.05 --data_path /path/to/imagenet/ --output_dir /output_dir/
  ```

#### Noticed:
- accum_iter set to 2 for 4 gpus (normally 8).
- warm-up epoch is set to 10 for 100 epochs and 40 for all other epoch settings.
- Note that min_lr is not set in the repo's command and is not explained in the paper.
- Code uses as many gpus as there are available, hence, set CUDA_VISIBLE_DEVICES.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 run_pretrain.py --epochs 100 --batch_size 256 --warmup_epochs 10 --world_size 4 --accum_iter 2 
```



## Steps

- Simple guide: [https://chat.openai.com/share/30777d71-4944-41ad-80a3-17dfca5bac7a](https://chat.openai.com/share/30777d71-4944-41ad-80a3-17dfca5bac7a)

### Step 1: Add FreezeOut Specific Variables

Add FreezeOut specific variables like `active`, `layer_index`, `max_j`, and `lr_ratio` to the layers you want to freeze. This is similar to what you did in the Wide ResNet model.

### Step 2: Modify the Optimizer

Modify the optimizer to take into account the `lr_ratio` for each layer, similar to how it's done in the Wide ResNet model.

### Step 3: Update Learning Rate

Implement a function to update the learning rate and freeze layers based on the `max_j` and `j` (iteration count), similar to the `update_lr` function in the Wide ResNet model. Both freezeout and localmim uses iteration count as steps for cosine annealing.

### Step 4: Modify the Forward Pass

Modify the forward pass to check if a layer is active or not. If it's not active, detach its gradients.

Here's a simplified example focusing on the key parts:

```
import numpy as np

scale_fn = {'linear':lambda x: x, 'squared': lambda x: x**2, 'cubic': lambda x: x**3}

# ... (existing code)

class BlockWithFreezeOut(Block):  # Assuming Block is your existing Block class
    def __init__(self, *args, layer_index, **kwargs):
        super(BlockWithFreezeOut, self).__init__(*args, **kwargs)
        self.active = True
        self.layer_index = layer_index
        self.lr_ratio = scale_fn['cubic'](0.1 + (1 - 0.1) * float(self.layer_index) / 100)  # Replace 100 with total layers
        self.max_j = 100 * 1000 * self.lr_ratio  # Replace 100 with total epochs

# ... (existing code)

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedAutoencoderViT, self).__init__(*args, **kwargs)
  
        # Replace existing blocks with BlockWithFreezeOut
        self.blocks = nn.ModuleList([
            BlockWithFreezeOut(*block_args, layer_index=i) for i, block_args in enumerate(self.blocks)
        ])
  
        # Initialize optimizer with lr_ratio
        self.optim = optim.SGD([{'params': m.parameters(), 'lr': m.lr_ratio} for m in self.modules() if hasattr(m, 'active')],  
                         nesterov=True, momentum=0.9, weight_decay=1e-4)
  
        self.j = 0  # Iteration counter

    def update_lr(self):
        # Similar to update_lr in Wide ResNet
        # ...

    def forward(self, x):
        # Modify to include active check
        for block in self.blocks:
            if block.active:
                x = block(x)
            else:
                x = block(x).detach()
        # ... (rest of the forward logic)
```

This is a simplified example and you'll need to adapt it to fit into your existing MIM ViT codebase. The key is to add the FreezeOut logic to the layers you want to freeze and then manage their state during training.

## Important details about Freezeout

- To pretrain MIM cosine lr scheduler has been used, so lars.py is not used at all:

  ![1693197965828](image/README/1693197965828.png)
- Fine tune settings are:

  ![1693198012042](image/README/1693198012042.png)
- Freezeout uses iteration wise cosine annealing (rather than epoch-wise), line of proof:

  ```
  # A simple dummy variable that indicates we are using an iteration-wise
          # annealing scheme as opposed to epoch-wise. 
          self.lr_sched = {'itr':0}
  ```

- The code assumes 1000 iterations per epoch and assigns the following max iterations per layer:  `m.max_j =self.epochs*1000*m.lr_ratio` (WRN.py). This might be a bad assumption. If you scale the learning rate linearly with the batch size, max_j iterations can be set by epochs/batch_size rather than a constant. Freezeout might be keeping initial lr and batch size independent, which might be the reason for this 1000.
- The learning rate is set to 0.1 by default for freeezeout, the cosine annealing step uses hence 0.05 (div by 2): `m.lr = 1e-1/m.lr_ratio if self.scale_lr else 1e-1`. Also cosine annealing max iterations are adjusted uniquely for each layer:

  ```
  self.optim.param_groups[i]['lr'] = (0.05/m.lr_ratio)*(1+np.cos(np.pi*self.j/m.max_j)) if self.scale_lr else 0.05 * (1+np.cos(np.pi*self.j/m.max_j)
  ```

## Important details about LocalMIM

- NOTE: Weight decay is applied selectively to non-bias or normalization layers with: `optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)`.
- NOTE: A different decay is applied during fine tuning: `param_groups_lrd`
- NOTE: Gradient clipping is only applied during fine tuning, not pretraining. Only amp scaling is applied during pretraining.
- NOTE: For pretraining, accum_iter: Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
- NOTE: Gradient Updates: Make sure that setting custom parameter groups doesn't interfere with how gradients are updated. You've mentioned that gradients will be accumulated; ensure that's compatible with your custom parameter groups.
- NOTE: min_lr is 0 based on freeze_out(j/max_j becomes 1 and layer is frozen after that)
- NOTE: blr is set already, you have to set batch_size to the same number as localmim paper.
- NOTE: the cls token serves as global image context summarizer, but here (MIM) it does not have a clear purpose (can be frozen)
- NOTE as the transformer has skip connection, the first input will flow till the end untouched.
- NOTE localmim and freezeout both use epoch count for cosine annealing.




## Important questions

- The method would benefit from freezing the decoder (4 times usage in a forward pass), but it will be used less as we freeze stages. Freezing the decoder together with the encoder should be kept as an ablation study, as it might behave unexpectedly.
