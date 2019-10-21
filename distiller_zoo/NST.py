from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class NSTLoss(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(NSTLoss, self).__init__()
        pass

    def forward(self, g_s, g_t):
        return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def nst_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        # set full_loss as False to avoid unnecessary computation
        full_loss = True
        if full_loss:
            return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean()
                    - 2 * self.poly_kernel(f_s, f_t).mean())
        else:
            return self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res