from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDSVD(nn.Module):
    """
    Self-supervised Knowledge Distillation using Singular Value Decomposition
    original Tensorflow code: https://github.com/sseung0703/SSKD_SVD
    """
    def __init__(self, k=1):
        super(KDSVD, self).__init__()
        self.k = k

    def forward(self, g_s, g_t):
        v_sb = None
        v_tb = None
        losses = []
        for i, f_s, f_t in zip(range(len(g_s)), g_s, g_t):

            u_t, s_t, v_t = self.svd(f_t, self.k)
            u_s, s_s, v_s = self.svd(f_s, self.k + 3)
            v_s, v_t = self.align_rsv(v_s, v_t)
            s_t = s_t.unsqueeze(1)
            v_t = v_t * s_t
            v_s = v_s * s_t

            if i > 0:
                s_rbf = torch.exp(-(v_s.unsqueeze(2) - v_sb.unsqueeze(1)).pow(2) / 8)
                t_rbf = torch.exp(-(v_t.unsqueeze(2) - v_tb.unsqueeze(1)).pow(2) / 8)

                l2loss = (s_rbf - t_rbf.detach()).pow(2)
                l2loss = torch.where(torch.isfinite(l2loss), l2loss, torch.zeros_like(l2loss))
                losses.append(l2loss.sum())

            v_tb = v_t
            v_sb = v_s

        bsz = g_s[0].shape[0]
        losses = [l / bsz for l in losses]
        return losses

    def svd(self, feat, n=1):
        size = feat.shape
        assert len(size) == 4

        x = feat.view(-1, size[1], size[2] * size[2]).transpose(-2, -1)
        u, s, v = torch.svd(x)

        u = self.removenan(u)
        s = self.removenan(s)
        v = self.removenan(v)

        if n > 0:
            u = F.normalize(u[:, :, :n], dim=1)
            s = F.normalize(s[:, :n], dim=1)
            v = F.normalize(v[:, :, :n], dim=1)

        return u, s, v

    @staticmethod
    def removenan(x):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return x

    @staticmethod
    def align_rsv(a, b):
        cosine = torch.matmul(a.transpose(-2, -1), b)
        max_abs_cosine, _ = torch.max(torch.abs(cosine), 1, keepdim=True)
        mask = torch.where(torch.eq(max_abs_cosine, torch.abs(cosine)),
                           torch.sign(cosine), torch.zeros_like(cosine))
        a = torch.matmul(a, mask)
        return a, b
