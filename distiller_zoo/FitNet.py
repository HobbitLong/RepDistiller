from __future__ import print_function

import torch.nn as nn


class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss
