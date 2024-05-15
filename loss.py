import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, logits=False, reduce=True, num_classes=9):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)  # num_classesをクラス数に設定
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha] * num_classes)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # alphaをtargetsのデバイスに移動
        if self.alpha.device != targets.device:
            self.alpha = self.alpha.to(targets.device)
        
        # デバッグ用: targetsの値をチェック
        if torch.any(targets >= len(self.alpha)):
            raise ValueError("targets contains values out of range for alpha")

        # alphaをtargetsの形状にブロードキャスト
        alpha_t = self.alpha[targets.data.view(-1).long()].view_as(targets)
        
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss