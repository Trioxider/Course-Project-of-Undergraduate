import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: 正类权重（控制正负样本不平衡）
        gamma: 聚焦参数（控制对困难样本的关注）
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits，大小为 [batch_size]
        targets: 标签，大小为 [batch_size]，0或1
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # [batch_size]
