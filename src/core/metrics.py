from typing import Tuple

import torch.tensor
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional


class FAFNCurve(pl.metrics.Metric):
    """
    This is the metric for calculating FA/FR curve.
    It assumes binary "guessed/not guessed" classification
    """
    def __init__(self, dist_sync_on_step=False, compute_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        self.add_state("probas", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.int32), dist_reduce_fx="cat")

    def update(self, probas: torch.Tensor, target: torch.Tensor) -> None:
        """
        Save probas of right keyword
        :param preds: torch.Tensor of size (batch_size, n_keywords) of probabilities of each keyword and no keyword
        :param target: torch.Tensor of size (batch_size) of keyword_ids (0 for no keyword)
        """
        self.target = torch.cat([self.target, target != 0])

        # Calculate the probability of wrong guess:
        probas[(target != 0)][:, 0] = 1 - probas[torch.arange(len(probas)), target][target != 0]
        probas[:, 1] = 1 - probas[:, 0]
        probas = probas[:, 1]
        self.probas = torch.cat([self.probas, probas], dim=0)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Copmute FA/FR for the whole set
        :return: tuple of (FA, FR, thresholds)
        """
        FA, tpr, thresholds = pl.metrics.functional.classification.roc(self.probas, self.target)
        FR = 1 - tpr
        return FA, FR, thresholds
