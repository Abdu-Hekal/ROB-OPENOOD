from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class JacNormPostprocessor(BasePostprocessor):
    """
    OOD postprocessor using the Jacobian norm of the model's predicted logit.
    Higher gradient norm indicates more likely OOD; we use the negative norm as the confidence score (ID > OOD).
    """
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    def postprocess(self, net: nn.Module, data: Any):
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
        # use negative norms so that lower gradient (ID) gives higher score
        conf = -norms
        return preds, conf 