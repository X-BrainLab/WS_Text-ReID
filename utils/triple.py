from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import norm

import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.K = num_instances
        self.use_semi = use_semi

    def forward(self, inputs, targets):
        n = inputs.size(0)
        P = n/self.K
        P = int(P)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if self.use_semi:
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i*self.K+j][mask[i*self.K+j] == 0]
                    for pair in range(j+1, self.K):
                        ap = dist[i*self.K+j][i*self.K+pair]

                        dist_ap.append(ap)
                        dist_an.append(neg_examples.min())
        else:
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max())
                dist_an.append(dist[i][mask[i] == 0].min())
        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
#
#
# class TripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None):
#         self.margin = margin
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an