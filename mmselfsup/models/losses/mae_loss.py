# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MAEReconstructionLoss(BaseModule):
    """Loss function for MAE.

    Compute the loss in masked region.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MAE Loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()

        return loss


@MODELS.register_module()
class MAEReconstructionLossWithIgnoreIndex(BaseModule):
    """Loss function for MAE.

    Compute the loss in masked region.
    """

    def __init__(self, ignore_index=255, reduction='mean') -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, non_255_mask:torch.Tensor) -> torch.Tensor:
        """Forward function of MAE L2 Loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.
            non_255_mask (torch.Tensor): The non_255 mask which indicates which token doesn't contain 255 value

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        # non_255_mask = (target == self.ignore_index).any(dim=2)
        # non_255_mask = (~non_255_mask).float()
        # diff = input.squeeze(-1) - target
        # if (target == 255).any():
        #     print('found 255 target value')
        loss = (pred - target)**2
        # Intersection between mask and non_255_mask
        mask = mask * non_255_mask

        # if torch.isnan(pred).any():
        #     print('Found nan in pred')

        # if torch.isnan(target).any():
        #     print('Found nan in target')
        loss = loss.mean(dim=-1)
        # non_nan_mask = (~torch.isnan(loss)).float()
        # convert nan containing losses to 0
        # loss = torch.nan_to_num(loss, nan=0)
        loss = (loss * mask).sum() / mask.sum()

        return loss


@MODELS.register_module()
class MAEL1ReconstructionLossWithIgnoreIndex(BaseModule):
    """Loss function for MAE.

    Compute the loss in masked region.
    """

    def __init__(self, ignore_index=255, reduction='mean') -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, non_255_mask:torch.Tensor) -> torch.Tensor:
        """Forward function of MAE L1 Loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.
            non_255_mask (torch.Tensor): The non_255 mask which indicates which token doesn't contain 255 value

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        # non_255_mask = (target == self.ignore_index).any(dim=2)
        # non_255_mask = (~non_255_mask).float()
        # diff = input.squeeze(-1) - target
        # if (target == 255).any():
        #     print('found 255 target value')
        loss = torch.abs(pred - target)
        # Intersection between mask and non_255_mask
        mask = mask * non_255_mask

        # if torch.isnan(pred).any():
        #     print('Found nan in pred')

        # if torch.isnan(target).any():
        #     print('Found nan in target')
        loss = loss.mean(dim=-1)
        # non_nan_mask = (~torch.isnan(loss)).float()
        # convert nan containing losses to 0
        # loss = torch.nan_to_num(loss, nan=0)
        loss = (loss * mask).sum() / mask.sum()

        return loss
