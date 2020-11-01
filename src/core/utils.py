import collections

import random
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import audiomentations as aud
import hydra
from omegaconf import DictConfig

import core


def fix_seeds(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


class PadCollator:
    def __init__(self, padding_value=-999.):
        self.padding_value = padding_value

    def __call__(self, batch):
        spectrograms, keyword_ids = zip(*batch)

        lengths = torch.tensor([len(s) for s in spectrograms])
        keyword_ids = torch.tensor(keyword_ids)
        spectrograms = nn.utils.rnn.pad_sequence([s.transpose(0, 1) for s in spectrograms],
                                                 batch_first=True, padding_value=self.padding_value)

        return spectrograms, lengths, keyword_ids


def get_class_weights(dataset: core.datasets.SPEECHCOMMANDS):
    """
    Get class for imbalanced dataset
    :param dataset: dataset with get_label to get label of an item with index i
    :return:
    """
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    weights = np.zeros_like(labels, dtype=np.float32)
    label_counts = collections.Counter(labels)
    for value, count in label_counts.items():
        weights[labels == value] = 1 / count
    return weights


def get_split(dataset: core.datasets.SPEECHCOMMANDS, random_state: int, train_size: float, stratify: bool = False):
    """
    Get train and test indices for dataset
    :param dataset: torch.Dataset (or any object with length)
    :param random_state: random state for dataset
    :param train_size: fraction of indices to use for training
    :param stratify: whether to stratify by labels
    :return:
    """
    idxs = np.arange(len(dataset))
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    train_idx, test_idx = sklearn.model_selection.train_test_split(idxs, train_size=train_size,
                                                                   stratify=stratify and labels,
                                                                   random_state=random_state)
    return train_idx, test_idx


def get_transforms(transforms: DictConfig):
    """
    get all necessary transforms from config
    :param transforms: transforms from config
    :return: transforms composed into aud.Compose
    """
    if transforms is None:
        return None
    return aud.Compose([
        hydra.utils.instantiate(transform)
        for transform in transforms
    ])
