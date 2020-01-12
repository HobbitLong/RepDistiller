from __future__ import print_function

import torch
import torch.nn as nn


class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


# class Correlation(nn.Module):
#     """Similarity-preserving loss. My origianl own reimplementation 
#     based on the paper before emailing the original authors."""
#     def __init__(self):
#         super(Correlation, self).__init__()
#
#     def forward(self, f_s, f_t):
#         return self.similarity_loss(f_s, f_t)
#         # return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
#
#     def similarity_loss(self, f_s, f_t):
#         bsz = f_s.shape[0]
#         f_s = f_s.view(bsz, -1)
#         f_t = f_t.view(bsz, -1)
#
#         G_s = torch.mm(f_s, torch.t(f_s))
#         G_s = G_s / G_s.norm(2)
#         G_t = torch.mm(f_t, torch.t(f_t))
#         G_t = G_t / G_t.norm(2)
#
#         G_diff = G_t - G_s
#         loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
#         return loss
