from typing import Any
import torch
import torch.nn as nn
from copy import deepcopy
import os

from .base_postprocessor import BasePostprocessor

class VariancePostprocessor(BasePostprocessor):
    """
    OOD postprocessor using variance of penultimate layer features under input noise.
    ID inputs should have stable features under noise; OOD have higher variance.
    For each input x, generates N noisy samples x_i = x + eta_i,
    computes mean and variance of base score for chosen postprocessor,
    and returns a score based on the combination of mean and variance.
    """
    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args
        # number of noisy samples to generate
        self.num_samples = self.args.num_samples
        # standard deviation (magnitude) of Gaussian noise
        self.noise_magnitude = self.args.noise_magnitude
        # score postprocessor name 
        self.score_pp_name = self.args.score_postprocessor
        # instantiate the base postprocessor for score computation
        # determine project config root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_root = os.path.join(project_root, 'configs')
        # get the in-distribution dataset name
        id_data_name = config.dataset.name
        from openood.evaluation_api.postprocessor import get_postprocessor
        self.base_pp = get_postprocessor(config_root, self.score_pp_name, id_data_name)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # forward setup to the selected score postprocessor, if supported
        try:
            self.base_pp.setup(net, id_loader_dict, ood_loader_dict)
        except AttributeError:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # get base predictions and scores from selected postprocessor
        preds, base_score = self.base_pp.postprocess(net, data)
        # collect scores from noisy samples
        scores = [base_score]
        for _ in range(self.num_samples-1):
            noisy_data = data + torch.normal(mean=torch.zeros_like(data), std=self.noise_magnitude)
            _, score = self.base_pp.postprocess(net, noisy_data)
            scores.append(score)
        # stack scores: (num_samples, batch_size)
        score_stack = torch.stack(scores, dim=0)
        # compute mean across noisy samples: (batch_size,)
        score_mean = score_stack.mean(dim=0)
        # compute variance across noisy samples: (batch_size,)
        score_var = score_stack.var(dim=0, unbiased=False)
        # compute average pairwise distance of base scores
        N = score_stack.shape[0]
        diff = score_stack.unsqueeze(1) - score_stack.unsqueeze(0)
        abs_diff = diff.abs()
        avg_pairwise_dist = abs_diff.sum(dim=(0,1)) / (N*(N-1))
        # compute maximum distance from mean as proxy for radius of minimum enclosing ball
        dist_from_mean = (score_stack - score_mean.unsqueeze(0)).abs()
        radius = dist_from_mean.max(dim=0)[0]
        # compute multiple confidence measures:
        # 1. mean score
        # 2. inverse of std deviation of base score
        inv_std = 1.0/(torch.sqrt(score_var) + 1e-12)
        # 3. inverse of average pairwise distance
        inv_apd = 1.0/(avg_pairwise_dist + 1e-12)
        # 4. inverse of radius
        inv_radius = 1.0/(radius + 1e-12)
        # return predictions and list of confidence measures
        return preds, [score_mean, inv_std, inv_apd, inv_radius] 