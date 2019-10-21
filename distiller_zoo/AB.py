from __future__ import print_function

import torch
import torch.nn as nn


class ABLoss(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """
    def __init__(self, feat_num, margin=1.0):
        super(ABLoss, self).__init__()
        self.w = [2**(i-feat_num+1) for i in range(feat_num)]
        self.margin = margin

    def forward(self, g_s, g_t):
        bsz = g_s[0].shape[0]
        losses = [self.criterion_alternative_l2(s, t) for s, t in zip(g_s, g_t)]
        losses = [w * l for w, l in zip(self.w, losses)]
        # loss = sum(losses) / bsz
        # loss = loss / 1000 * 3
        losses = [l / bsz for l in losses]
        losses = [l / 1000 * 3 for l in losses]
        return losses

    def criterion_alternative_l2(self, source, target):
        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        return torch.abs(loss).sum()
