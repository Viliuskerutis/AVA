import torch
import torch.nn as nn


class MidpointDeviationLoss(nn.Module):
    """
    Custom loss function that penalizes predictions based on their deviation
    from the precomputed midpoint of the estimated price range.
    """

    def forward(self, y_pred, midpoints):
        # Penalize based on the absolute deviation from the midpoint
        deviation = torch.abs(y_pred - midpoints)

        # Mean deviation loss
        loss = torch.mean(deviation)
        return loss
