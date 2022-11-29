import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class mse_loss(torch.nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
    def forward(self,y,t):
        return 0.5 * np.sum(y, t)

class cross_entropy_loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(cross_entropy_loss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class cross_entropy_focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=0.2, reduction='mean'):
        super(cross_entropy_focal_loss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)  # [NHW]
        log_gt = torch.log(pt)

        if self.alpha is not None:
            alpha = self.alpha.gather(0, target.view(-1))  # [NHW]
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt) ** self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class PolyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor = None, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets, label_smoothing=self.label_smoothing, weight=self.weight)
        pt = F.one_hot(targets, outputs.size()[1]) * F.softmax(outputs, 1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()




