import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import pyworld as pw
import numpy as np
import pandas as pd

from src.metric.eer import compute_eer


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        yaml.dump(content, handle, default_flow_style=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self._logits = []
        self._labels = []
        self._eer = None
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        
        self._logits.clear()
        self._labels.clear()
        self._eer = None

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        if key != 'eer':
            self._data.total[key] += value * n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key]
        else:
            logits, labels = value
            self._logits.append(logits)
            self._labels.append(labels)
        
    def avg(self, key):
        if key != 'eer':
            return self._data.average[key]
    
        logits = np.concatenate(self._logits, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        
        bonafide_scores = logits[labels == 0]
        other_scores = logits[labels == 1]
        
        self._eer = compute_eer(bonafide_scores, other_scores)
        return self._eer

    def result(self):
        return dict(self._data.average).update({
            "eer": self.avg("eer") if self._eer is None else self._eer
        })

    def keys(self):
        keys = self._data.total.keys()
        keys.append("eer")


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
