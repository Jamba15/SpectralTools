import torch.nn.utils.prune as prune
from .spectraldense import Spectral

def prune_percentile(model, percentile):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, Spectral):
          parameters_to_prune.append((module, 'eigvals'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percentile,
    )