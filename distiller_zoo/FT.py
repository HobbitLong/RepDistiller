from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class FactorTransfer(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer, NeurIPS 2018"""
    def __init__(self, p1=2, p2=1):
        super(FactorTransfer, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, f_s, f_t):
        return self.factor_loss(f_s, f_t)

    def factor_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        if self.p2 == 1:
            return (self.factor(f_s) - self.factor(f_t)).abs().mean()
        else:
            return (self.factor(f_s) - self.factor(f_t)).pow(self.p2).mean()

    def factor(self, f):
        return F.normalize(f.pow(self.p1).mean(1).view(f.size(0), -1))
