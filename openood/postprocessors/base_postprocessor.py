from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm
import numpy as np  # added for multi-confidence support


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        """
        Perform inference over the data_loader, collecting predictions, labels,
        and one or multiple confidence scores returned by postprocess.
        """
        pred_list = []
        label_list = []
        # containers for confidence(s)
        single_conf = []
        multi_conf = None
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            # postprocess returns pred and conf; postprocessor may set metric_labels internally
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            label_list.append(label.cpu())
            # handle single or multiple confidences
            if isinstance(conf, (list, tuple)):
                if multi_conf is None:
                    multi_conf = [[] for _ in range(len(conf))]
                for i, c in enumerate(conf):
                    multi_conf[i].append(c.cpu())
            else:
                if multi_conf is not None:
                    raise ValueError("Inconsistent conf outputs: mixture of single and multiple confidences.")
                single_conf.append(conf.cpu())

        # convert to numpy
        pred_arr = torch.cat(pred_list).numpy().astype(int)
        label_arr = torch.cat(label_list).numpy().astype(int)
        if multi_conf is not None:
            # multiple confidence methods: stack into shape [N, num_methods]
            conf_arrs = [torch.cat(lst).numpy() for lst in multi_conf]
            conf_arr = np.stack(conf_arrs, axis=1)
        else:
            # single confidence vector
            conf_arr = torch.cat(single_conf).numpy()
        return pred_arr, conf_arr, label_arr
