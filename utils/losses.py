import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, mask=None):
        loss = (y_pred - y_true)**2
        if mask is not None:
            count = torch.sum(mask)
            masked_loss = loss * mask
            return torch.sum(masked_loss) / count
        return torch.mean(loss)
