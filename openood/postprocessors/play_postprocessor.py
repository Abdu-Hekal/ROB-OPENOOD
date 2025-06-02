from typing import Any
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor

class PlayPostprocessor(BasePostprocessor):
    """
    OOD postprocessor for testing randomized smoothing ideas.
    For each input x, generates N noisy samples x_i = x + eta_i,
    computes mean and average pairwise distance of raw outputs,
    and returns a score based on their ratio.
    """
    def __init__(self, config):
        super().__init__(config)
        args = config.postprocessor.postprocessor_args
        # initialize hooks, layer list, and streaming buffers
        self.hooks = []
        self.layer_names = []
        # streaming accumulators per layer
        self.sum1 = {}
        self.sum2 = {}
        # flag to enable streaming accumulation only during postprocess
        self.streaming = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # register streaming hooks on all layers
        for name, module in net.named_modules():
            self.layer_names.append(name)
            handle = module.register_forward_hook(self._stream_hook(name))
            self.hooks.append(handle)

    def _stream_hook(self, name):
        # streaming accumulation of feature moments
        def hook(module, inp, output):
            if not self.streaming:
                return
            f = output.detach().flatten(start_dim=1)  # (batch_size, D)
            # initialize buffers on first call
            if name not in self.sum1:
                B, D = f.shape
                device = f.device
                self.sum1[name] = torch.zeros((B, D), device=device)
                self.sum2[name] = torch.zeros((B, D), device=device)
            # update streaming sums
            self.sum1[name] += f
            self.sum2[name] += f * f
        return hook

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        torch.cuda.empty_cache()
        # enable streaming for this postprocess session
        self.streaming = True
        # clear streaming accumulators
        self.sum1.clear()
        self.sum2.clear()

        num_samples = 100
        noise_magnitude = 0.01
        # collect raw logits and inputs for clean and noisy samples
        inputs = [data]
        outputs = [net(data)]
        for _ in range(num_samples - 1):
            noise = torch.normal(mean=torch.zeros_like(data), std=noise_magnitude)
            noisy = data + noise
            inputs.append(noisy)
            outputs.append(net(noisy))
        # stack inputs and outputs across samples
        input_stack = torch.stack(inputs, dim=0)
        # (N, batch_size, num_classes)
        output_stack = torch.stack(outputs, dim=0)
        # compute input variance summary across noise
        var_input = input_stack.var(dim=0, unbiased=False)  # (batch_size, *input_dims)
        var_input_summary = var_input.flatten(start_dim=1).mean(dim=1)  # (batch_size,)
        N, batch_size, num_classes = output_stack.shape
        eps = 1e-12

        # predicted classes based on mean logits
        output_mean = output_stack.mean(dim=0)
        preds = output_mean.argmax(dim=1)

        # 5. ratio of variance between adjacent layers (streaming)
        var_summary = {}
        Nf = float(num_samples)
        for name in self.layer_names:
            sum1 = self.sum1[name]   # (batch_size, D)
            sum2 = self.sum2[name]
            # per-dimension variance
            mu = sum1 / Nf
            var_feat = sum2 / Nf - mu * mu  # (batch_size, D)
            var_summary[name] = var_feat.mean(dim=1)  # (batch_size,)
        metrics5 = [
            var_summary[self.layer_names[i+1]] / (var_summary[self.layer_names[i]] + eps)
            for i in range(len(self.layer_names) - 1)
        ]
        # Additional variance ratio metrics
        first = self.layer_names[0]
        last = self.layer_names[-1]
        # ratio of variance between first and last layer
        metric_full_ratio = var_summary[last] / (var_summary[first] + eps)
        # ratio of variance between input and output
        metric_input_output = (var_input_summary) / (var_summary[last] + eps)
        # collate all ratio metrics
        metrics = metrics5 + [metric_full_ratio, metric_input_output]
        # build labels for each metric
        labels = [f'var_ratio_l{i+1}_l{i+2}' for i in range(len(self.layer_names) - 1)]
        labels += [f'var_ratio_l1_l{len(self.layer_names)}', 'var_ratio_input_output']
        # store metric labels on the postprocessor
        self.metric_labels = labels
        # disable streaming after accumulation
        self.streaming = False
        return preds, metrics