import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
import collections
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.evaluation_api import Evaluator

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet
from openood.networks.palm_net import PALMNet
from openood.networks.t2fnorm_net import T2FNormNet
from openood.networks.ascood_net import ASCOODNet

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--postprocessor', default='msp')
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet200'])
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--wrapper-net',
                    type=str,
                    default=None,
                    choices=['ASCOODNet'])
parser.add_argument('--plot-score', type=lambda x: x.lower() in ('true','1','yes'), default=True, help='Enable plotting of score histograms (true/false).')
args = parser.parse_args()

root = args.root

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
}

try:
    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]
except KeyError:
    raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

# assume that the root folder contains subfolders each corresponding to
# a training run, e.g., s0, s1, s2
# this structure is automatically created if you use OpenOOD for train
if len(glob(os.path.join(root, 's*'))) == 0:
    raise ValueError(f'No subfolders found in {root}')

# iterate through training runs
all_metrics = []
for subfolder in sorted(glob(os.path.join(root, 's*'))):
    # load pre-setup postprocessor if exists
    if os.path.isfile(
            os.path.join(subfolder, 'postprocessors',
                         f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'postprocessors',
                             f'{postprocessor_name}.pkl'), 'rb') as f:
            postprocessor = pickle.load(f)
    else:
        postprocessor = None

    # load the pretrained model provided by the user
    if postprocessor_name == 'conf_branch':
        net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                            num_classes=num_classes)
    elif postprocessor_name == 'godin':
        backbone = model_arch(num_classes=num_classes)
        net = GodinNet(backbone=backbone,
                       feature_size=backbone.feature_size,
                       num_classes=num_classes)
    elif postprocessor_name == 'rotpred':
        net = RotNet(backbone=model_arch(num_classes=num_classes),
                     num_classes=num_classes)
    elif 'csi' in root:
        backbone = model_arch(num_classes=num_classes)
        net = CSINet(backbone=backbone,
                     feature_size=backbone.feature_size,
                     num_classes=num_classes)
    elif 'udg' in root:
        backbone = model_arch(num_classes=num_classes)
        net = UDGNet(backbone=backbone,
                     num_classes=num_classes,
                     num_clusters=1000)
    elif postprocessor_name in ['cider', 'reweightood']:
        backbone = model_arch(num_classes=num_classes)
        net = CIDERNet(backbone,
                       head='mlp',
                       feat_dim=128,
                       num_classes=num_classes)
    elif postprocessor_name == 'npos':
        backbone = model_arch(num_classes=num_classes)
        net = NPOSNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
    elif postprocessor_name == 'palm':
        backbone = model_arch(num_classes=num_classes)
        net = PALMNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
        postprocessor_name = 'mds'
    elif postprocessor_name == 't2fnorm':
        backbone = model_arch(num_classes=num_classes)
        net = T2FNormNet(backbone=backbone, num_classes=num_classes)
    else:
        net = model_arch(num_classes=num_classes)

    if args.wrapper_net is not None:
        net = eval(args.wrapper_net)(backbone=net)

    net.load_state_dict(
        torch.load(os.path.join(subfolder, 'best.ckpt'), map_location='cpu'))
    net.cuda()
    net.eval()

    evaluator = Evaluator(
        net,
        id_name=args.id_data,  # the target ID dataset
        data_root=os.path.join(ROOT_DIR, 'data'),
        config_root=os.path.join(ROOT_DIR, 'configs'),
        preprocessor=None,  # default preprocessing
        postprocessor_name=postprocessor_name,
        postprocessor=
        postprocessor,  # the user can pass his own postprocessor as well
        batch_size=args.
        batch_size,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=8)

    # load pre-computed scores if exist
    if os.path.isfile(
            os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl'),
                'rb') as f:
            scores = pickle.load(f)
        update(evaluator.scores, scores)
        print('Loaded pre-computed scores from file.')

    # save the postprocessor for future reuse
    if hasattr(evaluator.postprocessor, 'setup_flag'
               ) or evaluator.postprocessor.hyperparam_search_done is True:
        pp_save_root = os.path.join(subfolder, 'postprocessors')
        if not os.path.exists(pp_save_root):
            os.makedirs(pp_save_root)

        if not os.path.isfile(
                os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
            with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
                      'wb') as f:
                pickle.dump(evaluator.postprocessor, f,
                            pickle.HIGHEST_PROTOCOL)

    metrics = evaluator.eval_ood(fsood=args.fsood)
    all_metrics.append(metrics.to_numpy())

    if args.plot_score:
        # collect ID confidences
        _, id_conf, _ = evaluator.scores['id']['test']
        # collect OOD confidences per dataset
        ood_conf_map = {}
        for split in ['near', 'far']:
            for ds_name, ds_vals in evaluator.scores['ood'][split].items():
                if ds_vals is None:
                    continue
                ood_conf_map[ds_name] = ds_vals[1]
        # plot all distributions in one figure
        plt.figure()
        plt.hist(id_conf, bins=50, alpha=0.5, label='ID')
        for ds_name, ood_conf in ood_conf_map.items():
            plt.hist(ood_conf, bins=50, alpha=0.5, label=ds_name)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.title(f'ID vs OOD score histogram ({os.path.basename(subfolder)})')
        plot_dir = os.path.join(subfolder, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'hist_{postprocessor_name}.png'))
        plt.close()

    # save computed scores
    if args.save_score:
        score_save_root = os.path.join(subfolder, 'scores')
        if not os.path.exists(score_save_root):
            os.makedirs(score_save_root)
        with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
                  'wb') as f:
            pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)

# compute mean metrics over training runs
all_metrics = np.stack(all_metrics, axis=0)
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)

final_metrics = []
for i in range(len(metrics_mean)):
    temp = []
    for j in range(metrics_mean.shape[1]):
        temp.append(u'{:.2f} \u00B1 {:.2f}'.format(metrics_mean[i, j],
                                                   metrics_std[i, j]))
    final_metrics.append(temp)
df = pd.DataFrame(final_metrics, index=metrics.index, columns=metrics.columns)

if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    df.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'))
else:
    print(df)
