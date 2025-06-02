from typing import Any
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor

class RatioPostprocessor(BasePostprocessor):
    """
    OOD postprocessor computing ratio of feature activation magnitude between layers.
    For a single forward pass, it computes mean absolute activation of each module's output
    and returns ratio metrics between adjacent layers and between first and last layer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.hooks = []
        self.layer_names = []
        self.outputs = {}

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # register hooks on all modules to capture outputs
        for name, module in net.named_modules():
            self.layer_names.append(name)
            handle = module.register_forward_hook(self._hook(name))
            self.hooks.append(handle)

    def _hook(self, name):
        def hook(module, inp, output):
            # flatten output and compute mean absolute activation per sample
            f = output.detach().flatten(start_dim=1)  # (batch_size, D)
            summary = f.abs().mean(dim=1)  # (batch_size,)
            self.outputs[name] = summary
        return hook

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # clear previous outputs
        self.outputs.clear()
        # single forward pass
        output = net(data)
        # predicted classes
        preds = output.argmax(dim=1)
        eps = 1e-12
        metrics = []
        # compute ratio between adjacent layers
        for i in range(len(self.layer_names) - 1):
            name1 = self.layer_names[i]
            name2 = self.layer_names[i + 1]
            summary1 = self.outputs.get(name1)
            summary2 = self.outputs.get(name2)
            if summary1 is None or summary2 is None:
                raise ValueError(f'Missing output for layer {name1} or {name2}')
            metrics.append(summary2 / (summary1 + eps))
        # append raw output summary for each layer
        for name in self.layer_names:
            metrics.append(self.outputs[name])
            print(f'{name}: {self.outputs[name].shape}')
        # build metric labels
        ratio_labels = [f'ratio_l{i+1}_l{i+2}' for i in range(len(self.layer_names) - 1)]
        raw_labels = [f'output_l{i+1}' for i in range(len(self.layer_names))]
        labels = ratio_labels + raw_labels
        self.metric_labels = labels
        return preds, metrics 
    
