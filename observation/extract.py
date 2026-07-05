import torch
from torch import nn
from typing import Dict

# It will extract weight matrices from a model, and returning 2d tensors
def extract_weight_matrices(
    model:nn.Module,
    mode:str = "conv_linear"  # (conv_linear:only nn.conv2d and nn.linear) or (all:every module that has .weight attribute)
) -> Dict[str,torch.Tensor]:
    
    layers:Dict[str,torch.Tensor] = {}
    for name,module in model.named_modules():
        if mode=="conv_linear":
            if not isinstance(module,(nn.Linear,nn.Conv2d)):
                continue
        else:
            if not (hasattr(module,"weight") and module.weight is not None):
                continue

        W = module.weight.detach().clone()  #type:ignore
        if W.dim() == 4:
            W = W.flatten(1)
        if W.dim() < 2:
            continue

        layers[name] = W
    return layers

def print_layer_summary(
    layers:Dict[str,torch.Tensor]
) -> None:
    print(f"\n{'Layer Name':<45} {'Shape':<20} {'Params':>10}")
    print("-"*80)
    total = 0
    for name,W in layers.items():
        params = W.numel()
        total += params
        print(f"{name:<45} {str(tuple(W.shape)):20} {params:>10,}")
    print("-"*80)
    print(f"{'TOTAL':<45} {'':<20} {total:>10,}\n")