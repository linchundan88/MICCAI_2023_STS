from torch import nn as nn

from SegLoss.losses_pytorch.dice_loss import IoULoss
from SegLoss.losses_pytorch.hausdorff import HausdorffDTLoss


class complex_criterion():
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        self.hd_loss = HausdorffDTLoss()

    def __call__(self, pred, label):
        return self.bce_loss(pred, label) \
            # + 0.5 * self.hd_loss(pred, label)
        # + 0.5 * self.iou_loss(pred, label)
