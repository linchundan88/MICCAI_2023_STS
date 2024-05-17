import numpy as np
import torch
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou
from monai.metrics.utils import do_metric_reduction
from monai.metrics.utils import get_mask_edges, get_surface_distance
from monai.metrics import CumulativeIterationMetric

class HausdorffScore(CumulativeIterationMetric):
    """
    Modify MONAI's HausdorffDistanceMetric for Kaggle UW-Madison GI Tract Image Segmentation

    """

    def __init__(
            self,
            reduction="mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return compute_hausdorff_score(pred, gt)

    def aggregate(self):
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        """
        data = self.get_buffer()
        # do metric reduction
        f, _ = do_metric_reduction(data, self.reduction)
        return f


# hausdorff
def compute_hausdorff_score(pred, gt):
    y = gt.float().to("cpu").numpy()
    y_pred = pred.float().to("cpu").numpy()

    # hausdorff distance score
    batch_size, n_class = y_pred.shape[:2]
    spatial_size = y_pred.shape[2:]
    max_dist = np.sqrt(np.sum([l ** 2 for l in spatial_size]))
    hd_score = np.empty((batch_size, n_class))
    for b, c in np.ndindex(batch_size, n_class):
        hd_score[b, c] = 1 - compute_directed_hausdorff(y_pred[b, c], y[b, c], max_dist)

    return torch.from_numpy(hd_score)


def compute_directed_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    if np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0:
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()

    if dist > max_dist:
        return 1.0
    return dist / max_dist


hausdorff_metric = HausdorffScore(reduction="mean")


# def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
#     y_true = y_true.to(torch.float32)
#     y_pred = (y_pred>thr).to(torch.float32)
#     inter = (y_true*y_pred).sum(dim=dim)
#     den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
#     dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
#     return dice
#
#
# def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
#     y_true = y_true.to(torch.float32)
#     y_pred = (y_pred>thr).to(torch.float32)
#     inter = (y_true*y_pred).sum(dim=dim)
#     union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
#     iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
#     return iou


def scores_coef(y_true, y_pred, thr=0.5):
    y_pred = y_pred[None, None, :, :]   #(H, W)  - >  (N, C, H, W)  monai 1.3, otherwise compute_dice will get wrong results.
    y_true = y_true[None, None, :, :]

    y_pred = torch.ge(y_pred, thr).to(torch.int)

    dice_score = compute_dice(y_pred, y_true)
    iou_score = compute_iou(y_pred, y_true)
    hausdorff_metric(y_pred=y_pred, y=y_true)
    hausdorff_score = hausdorff_metric.aggregate().item()

    score = float(0.4*dice_score+0.3*iou_score+0.3*(hausdorff_score))

    return score


