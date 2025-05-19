from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.func import jacrev


from .base_postprocessor import BasePostprocessor


class DeepfoolPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DeepfoolPostprocessor, self).__init__(config)
        args = config.postprocessor.postprocessor_args
        self.num_classes = args.num_classes
        self.overshoot = args.overshoot
        self.max_iter = args.max_iter

    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        batch_size = data.shape[0]
        preds = []
        confs = []
        device = data.device
        dtype = data.dtype

        for i in range(batch_size):
            image = data[i]
            r_tot, iter_count, label_init, label_final, pert_image = self.deepfool(image, net)
            # score is L2 norm of the total perturbation
            score = float(torch.norm(r_tot).item())
            preds.append(int(label_init))
            confs.append(float(score))

        pred_tensor = torch.tensor(preds, device=device, dtype=torch.int64)
        conf_tensor = torch.tensor(confs, device=device, dtype=torch.float32)
        return pred_tensor, conf_tensor

    def deepfool(self, image, net, num_classes=None, overshoot=None, max_iter=None):
        """Vectorized DeepFool attack using functorch.jacrev and pure torch operations."""
        device = image.device
        dtype = image.dtype
        num_classes = num_classes or self.num_classes
        overshoot = overshoot or self.overshoot
        max_iter = max_iter or self.max_iter

        # Prepare image and model
        image = image.clone().to(device)
        net = net.to(device).eval()

        # Initial prediction and ordering
        x0 = image.unsqueeze(0)
        fs0 = net(x0)
        _, I = fs0[0].sort(descending=True)
        I = I[:num_classes]
        label = I[0].item()

        pert_image = image.clone()
        r_tot = torch.zeros_like(image)
        loop_i = 0
        k_i = label

        while k_i == label and loop_i < max_iter:
            x = pert_image.unsqueeze(0).requires_grad_(True)
            fs = net(x)[0, I]  # logits for top classes

            # Compute gradients for all target logits at once
            grads = jacrev(lambda inp: net(inp)[0, I])(x)
            grads = grads.reshape(num_classes, *x.shape[1:])

            grad_orig = grads[0]
            w_k = grads[1:] - grad_orig.unsqueeze(0)
            fs_vals = fs
            f_diffs = fs_vals[1:] - fs_vals[0]

            w_k_flat = w_k.view(w_k.shape[0], -1)
            norms = w_k_flat.norm(dim=1)
            pert_k = f_diffs.abs() / norms

            min_pert, min_idx = pert_k.min(dim=0)
            w = w_k[min_idx]

            # Compute perturbation
            r_i = (min_pert + 1e-4) * w / w.norm()
            r_tot = r_tot + r_i

            # Apply perturbation
            pert_image = image + (1 + overshoot) * r_tot

            # Update label
            with torch.no_grad():
                k_i = net(pert_image.unsqueeze(0)).argmax(dim=1).item()
            loop_i += 1

        r_tot = (1 + overshoot) * r_tot
        return r_tot, loop_i, label, k_i, pert_image