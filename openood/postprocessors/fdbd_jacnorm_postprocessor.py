from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from .fdbd_postprocessor import fDBDPostprocessor


class FdbdJacNormPostprocessor(BasePostprocessor):
    """
    Composite OOD score: time to boundary = fdbd_distance / jacobian_norm(feature).
    fdbd_distance from fDBDPostprocessor, jacobian_norm on penultimate features.
    """
    def __init__(self, config):
        super().__init__(config)
        # underlying fdbd postprocessor
        self.fdbd = fDBDPostprocessor(config)
        self.args = config.postprocessor.postprocessor_args

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # setup fdbd components (train mean, denominators)
        self.fdbd.setup(net, id_loader_dict, ood_loader_dict)

    def postprocess(self, net: nn.Module, data: Any):
        # get fdbd distance score (no_grad inside)
        idx, fdbd_score = self.fdbd.postprocess(net, data)
        # enable gradient tracking on input
        data.requires_grad_(True)
        # forward pass
        logits = net(data)
        # predicted classes
        preds = logits.argmax(dim=1)
        # select the logit corresponding to each prediction
        selected = logits[torch.arange(logits.size(0)), preds]
        # compute gradient of the selected logits w.r.t. input
        grads = torch.autograd.grad(selected.sum(), data)[0]
        # compute L2 norm of gradients for each sample
        norms = grads.view(grads.size(0), -1).norm(p=2, dim=1)
        # time-to-boundary: distance / speed (avoid div0)
        score = fdbd_score-(norms) #fdbd_score / (norms + eps)
        return preds, score 