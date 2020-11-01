import os
from pathlib import Path
from typing import Sequence, Optional

import torch.utils.data as torchdata
import torchaudio


class SPEECHCOMMANDS(torchdata.Dataset):
    """
    Wrapper for torchaudio.datasets.SPEECHCOMMANDS with predefined keywords
    :param root: Path to the directory where the dataset is found or downloaded.
    :param url: The URL to download the dataset from, or the type of the dataset to dowload. Allowed type values are
    "speech_commands_v0.01" and "speech_commands_v0.02" (default: "speech_commands_v0.02")
    :param keywords: List of keywords that will correspond to label 1
    :param download: Whether to download the dataset if it is not found at root path. (default: False)
    :param transforms: audiomentations transform object
    """
    def __init__(self, root: str, url: str, keywords: Sequence[str], download: bool = False, transforms=None) -> None:
        root = Path(root)
        if download and not root.exists():
            root.mkdir()
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, url=url, download=download)
        self.keywords = list(keywords)
        self.transforms = transforms

    def __getitem__(self, idx):
        (waveform, sample_rate, label, speaker_id, utterance_number) = self.dataset[idx]
        if self.transforms is not None:
            waveform = self.transforms(samples=waveform, sample_rate=sample_rate)
        try:
            keyword_id = self.keywords.index(label) + 1
        except ValueError:
            keyword_id = 0  # Not a keyword
        return waveform, keyword_id  # May be not waveform already

    def __len__(self) -> int:
        return len(self.dataset)

    def get_label(self, idx) -> int:
        """
        Get label only from the dataset
        :param idx: object index
        :return: object keyword_id
        """
        filepath = self.dataset._walker[idx]
        relpath = os.path.relpath(filepath, self.dataset._path)
        label, filename = os.path.split(relpath)
        try:
            keyword_id = self.keywords.index(label) + 1
        except ValueError:
            keyword_id = 0  # Not a keyword
        return keyword_id
