import torch
import numpy as np
import torchvision.models as models
from typing import Dict

# model = models.resnet18(pretrained=True)

def extract_weight_matrix(model, std_matrix=False) -> Dict:
    layers = {}
    for name, module in model.named_modules():
        # Mode 1: only Conv + Linear
        if not std_matrix:
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                W = module.weight.data

                if W.dim() == 4:
                    W = W.flatten(1)

                layers[name] = W

        # Mode 2: all weight-bearing modules
        else:
            if hasattr(module, "weight") and module.weight is not None:
                W = module.weight.data
                if W.dim() == 4:
                    W = W.flatten(1)
                layers[name] = W
    return layers
# You can edit this code as you need to see what you want to know of the given matrix
def print_Imp_Info(layers:Dict):
    for layer in layers:
        np_arr = np.array(layers[layer])
        print(f"Layer Name:{layer} | Layer Size:{np_arr.size} | Shape:{tuple(layers[layer].shape)}")

def alignment_score(W):
    # Your resultant = mean row direction
    r_mean = W.mean(dim=0)
    r_mean = r_mean/r_mean.norm()

    # SVD resultant = top singulat vector
    _, _, Vt = torch.linalg.svd(W,full_matrices=False)
    r_svd = Vt[0]

    # Cosine similarity of each row with resultant
    W_norm = W / (W.norm(dim=1,keepdim=True) + 1e-8)
    cos_mean = (W_norm @ r_mean).abs()
    cos_svd = (W_norm @ r_svd).abs()

    return {
        "mean_cos_mean_resultant":cos_mean.mean().item(),
        "mean_cos_svd_resultant":cos_svd.mean().item(),
        "std_cos":cos_svd.std().item(),

    }

def rank1_energy(W):
    S = torch.linalg.svdvals(W)
    return (S[0]**2 / (S**2).sum()).item()

if __name__ == "__main__":
    test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    trainable_layers = extract_weight_matrix(test_model)
    for name, W in trainable_layers.items():
        scores = alignment_score(W)
        e = rank1_energy(W)
        print(f"{name:40s} | align={scores['mean_cos_svd_resultant']:.3f} | rank1_energy={e:.3f}")