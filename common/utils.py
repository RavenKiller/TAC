import os
import glob
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def get_checkpoint_id(ckpt_path: str):
    """Get the index number in the ckpt_path."""
    ckpt_path = os.path.basename(ckpt_path)
    nums = ckpt_path.split(".")
    if len(nums) > 1:
        return int(nums[-2])
    return None


def poll_checkpoint_folder(checkpoint_folder: str, previous_ckpt_ind: int):
    """Sort checkpoints in checkpoint_folder by the index number in the filename. Then return the {previous_ckpt_ind+1}-th chekpoint filename."""
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(
            lambda x: (os.path.isfile(x) and "ckpt" in x),
            glob.glob(checkpoint_folder + "/*"),
        )
    )
    models_paths.sort(key=lambda x: int(x.split(".")[-2]))
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    Implementation from https://github.com/AdeelH/pytorch-multi-class-focal-loss

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, d1, d2, ..., dK, C) --> (N * d1 * ... * dK, C)
            c = x.shape[-1]
            x = x.reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def cross_entropy_focal(
    x: Tensor,
    y: Tensor,
    alpha: Optional[Tensor] = None,
    gamma: float = 0.0,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> Tensor:
    """Toll function for focal loss."""
    criterion = FocalLoss(
        torch.tensor(alpha).to(x.device), gamma, reduction, ignore_index
    )
    return criterion(x, y)
