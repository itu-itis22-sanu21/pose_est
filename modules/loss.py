import torch
import torch.nn.functional as F

def l2_loss(predicted, target, mask, batch_size):
    """
    Mean squared error (L2) loss for keypoint heatmaps.
    Arguments:
        predicted: Predicted heatmaps, shape (B, C, H, W)
        target: Ground truth heatmaps, shape (B, C, H, W)
        mask: Mask for valid keypoints, shape (B, C, H, W)
        batch_size: Batch size
    Returns:
        Scalar loss tensor
    """
    loss = (predicted - target) * mask
    loss = (loss * loss) / 2 / batch_size
    return loss.sum()
