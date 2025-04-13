import torch.nn as nn


class CombinedLoss(nn.Module):
    """
    Combines two loss functions with a weighted alpha parameter.
    """

    def __init__(self, loss_fn1, loss_fn2, alpha):
        """
        Args:
            loss_fn1 (nn.Module): First loss function.
            loss_fn2 (nn.Module): Second loss function.
            alpha (float): Weight for the first loss function. The second loss function will have a weight of (1 - alpha).
        """
        super().__init__()
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.alpha = alpha

    def forward(self, y_pred, y_true, *args):
        """
        Compute the combined loss.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Combined loss value.
        """
        loss1 = self.loss_fn1(y_pred, y_true, *args)
        loss2 = self.loss_fn2(y_pred, y_true, *args)
        return self.alpha * loss1 + (1 - self.alpha) * loss2
