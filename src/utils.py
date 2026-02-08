import json
import os
import random

# from typing import Counter
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim as optim


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def get_config(path):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def dump_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def append_log(log_path, data: dict):
    with open(log_path, "a") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")


def mk_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks=[1, 10, 20]):
    """
    Compute Recall@k and nDCG@k metrics.

    Args:
        scores: Prediction scores tensor, shape (batch_size, num_items)
        labels: Ground truth indices tensor, shape (batch_size,)
        ks: List of k values to evaluate

    Returns:
        Dictionary with Recall@k and nDCG@k for each k.
    """
    metrics = {}

    # Compute rank of the ground truth item for each sample
    rank = (
        (-scores).argsort(dim=-1).argsort(dim=-1)[torch.arange(scores.size(0)), labels]
    )

    for k in sorted(ks, reverse=True):
        # Recall@k
        metrics["R%d" % k] = (sum(rank < k) / labels.size(0)).item()
        # nDCG@k
        metrics["N%d" % k] = torch.mean(
            torch.where(rank < k, 1 / torch.log2(rank + 2), torch.tensor(0.0))
        ).item()

    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }

    def sums(self, format_string="{}"):
        return {
            format_string.format(name): meter.sum for name, meter in self.meters.items()
        }

    def counts(self, format_string="{}"):
        return {
            format_string.format(name): meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
