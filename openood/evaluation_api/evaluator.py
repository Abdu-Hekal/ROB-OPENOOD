from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor

import yaml

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
        robustbench=False,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # wrap base model to work with certain postprocessors
        if postprocessor_name == 'react'or postprocessor_name =='neo_react':
            net = ReactNet(net)
        elif postprocessor_name in ['ash' ,'neo_ash', 'pro_ash']:
            net = ASHNet(net)
        elif 'scale' in postprocessor_name:
            net = ScaleNet(net)

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])
        self.postprocessor_name=postprocessor_name
        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    self.scores['id_labels'] = all_labels

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'Unknown data name {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        # unified OOD evaluation supporting multiple confidence methods
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        if self.metrics[task] is None:
            self.net.eval()
            # obtain ID predictions and confidences
            if self.scores['id']['test'] is None:
                print(f'Performing inference on {self.id_name} test set...', flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']
            # handle class-split ID if fsood
            if fsood:
                cs_preds, cs_confs, cs_gts = [], [], []
                for i, ds in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][ds] is None:
                        print(f'Performing inference on CSID set [{i+1}]: {ds}', flush=True)
                        tp, tc, tg = self.postprocessor.inference(
                            self.net, self.dataloader_dict['csid'][ds], progress)
                        self.scores['csid'][ds] = [tp, tc, tg]
                    tp, tc, tg = self.scores['csid'][ds]
                    cs_preds.append(tp)
                    cs_confs.append(tc)
                    cs_gts.append(tg)
                id_pred = np.concatenate([id_pred] + cs_preds)
                id_conf = np.concatenate([id_conf] + cs_confs, axis=0)
                id_gt = np.concatenate([id_gt] + cs_gts, axis=0)
            # get DataFrames for near and far splits with MultiIndex
            df_near = self._eval_ood([id_pred, id_conf, id_gt], ood_split='near', progress=progress)
            df_far = self._eval_ood([id_pred, id_conf, id_gt], ood_split='far', progress=progress)
            # override ACC column with classifier accuracy
            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(id_name)
            id_acc = self.metrics[f'{id_name}_acc']
            # set ACC for all rows and concatenate near/far DataFrames
            df_near['ACC'] = id_acc
            df_far['ACC'] = id_acc
            self.metrics[task] = pd.concat([df_near, df_far])
        else:
            print('Evaluation has already been done!')
        # display full metrics table, splitting by confidence method only if multiple
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            df = self.metrics[task]
            # identify unique confidence methods
            confs = df.index.get_level_values('conf').unique()
            if len(confs) > 1:
                # multiple methods: print a table per method
                for conf in confs:
                    print(f"=== {conf or 'default'} ===")
                    df_conf = df.xs(conf, level='conf')
                    print(df_conf)
                    print()
            else:
                # single method: drop conf level and print full table
                df_single = df.droplevel('conf')
                print(df_single)
        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True):
        """
        Evaluate OOD for a given split, supporting multiple confidence methods.
        Returns:
            metrics_arr: np.ndarray of shape [num_rows, 5] (rows include per-dataset and mean)
            index_list: List[str] of length num_rows for DataFrame index
        """
        print(f'Processing {ood_split} ood...', flush=True)
        id_pred, id_conf, id_gt = id_list
        # collect records as (dataset, method, metrics_array)
        records = []
        # Iterate through each OOD dataset
        for dataset_name, ood_dl in self.dataloader_dict['ood'][ood_split].items():
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...', flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [ood_pred, ood_conf, ood_gt]
            else:
                print(f'Inference has been performed on {dataset_name} dataset...', flush=True)
                ood_pred, ood_conf, ood_gt = self.scores['ood'][ood_split][dataset_name]
            # label OOD as -1
            ood_gt = -1 * np.ones_like(ood_gt)
            # combine ID and OOD
            pred = np.concatenate([id_pred, ood_pred], axis=0)
            conf = np.concatenate([id_conf, ood_conf], axis=0)
            label = np.concatenate([id_gt, ood_gt], axis=0)
            # compute metrics
            if conf.ndim == 1:
                # single confidence method
                m = compute_all_metrics(conf, label, pred)
                records.append((dataset_name, '', m))
                # print metric label for this method
                print(f"--- {dataset_name} ---")
                self._print_metrics(m)
            else:
                # multiple confidence methods
                for i in range(conf.shape[1]):
                    m = compute_all_metrics(conf[:, i], label, pred)
                    # determine method label
                    if hasattr(self.postprocessor, 'metric_labels'):
                        method = self.postprocessor.metric_labels[i]
                    else:
                        method = f"c{i}"
                    records.append((dataset_name, method, m))
                    # print metric label for this method
                    print(f"--- {dataset_name}_{method} ---")
                    self._print_metrics(m)
        # compute mean across datasets for each method
        print('Computing mean metrics...', flush=True)
        if conf.ndim == 1:
            # mean over datasets for single method
            arrs = [m for _,_,m in records]
            mean_m = np.mean(arrs, axis=0)
            records.append((f'{ood_split}ood', '', mean_m))
        else:
            num_conf = conf.shape[1]
            for i in range(num_conf):
                # collect metrics of method i across all datasets
                if hasattr(self.postprocessor, 'metric_labels'):
                    method = self.postprocessor.metric_labels[i]
                else:
                    method = f'c{i}'
                arrs = [m for _,mth,m in records if mth == method]
                mean_m = np.mean(arrs, axis=0)
                records.append((f'{ood_split}ood', method, mean_m))
                # print metric label for this method
                print(f"--- {ood_split}ood_{method} ---")
                self._print_metrics(mean_m)
        # build DataFrame
        data, idx = [], []
        for ds, method, m in records:
            # convert metric vector to numpy array, then scale
            arr = np.array(m)
            data.append(arr * 100)
            idx.append((ds, method))
        df = pd.DataFrame(data,
                          index=pd.MultiIndex.from_tuples(idx, names=['dataset','conf']),
                          columns=['FPR@95','AUROC','AUPR_IN','AUPR_OUT','ACC'])
        return df

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out, _] = metrics

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True
        self._save_hyperparameters_to_file(hyperparam_combination[final_index], max_auroc)

    def _save_hyperparameters_to_file(self, hyperparams, max_auroc):
        file_path = "ASPhyperparameters.yml"
        # Convert numpy types to native Python types for compatibility with YAML
        def convert_to_python_type(value):
            if isinstance(value, np.generic):  
                return value.item()
            elif isinstance(value, list): 
                return [convert_to_python_type(v) for v in value]
            return value
        hyperparams = convert_to_python_type(hyperparams)
        max_auroc = convert_to_python_type(max_auroc)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = yaml.safe_load(file) or {}
        else:
            data = {}
        # Update the data for the current postprocessor
        if self.postprocessor_name not in data:
            data[self.postprocessor_name] = {}
        data[self.postprocessor_name][self.id_name] = {
            "hyperparameters": hyperparams,
            "max_auroc": max_auroc
        }

        with open(file_path, "w") as file:
            yaml.dump(data, file)
        print(f'Hyperparameters for {self.postprocessor_name} saved to {file_path}')

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results