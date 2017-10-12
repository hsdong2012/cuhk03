from __future__ import print_function
import argparse
import numpy as np
import random
import torch

import torch.nn as nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, camera_a, camera_b, P, K):
	
        assert camera_a.size() == camera_b.size(), "Input sizes between camera_a and camera_b must be equal."
        assert camera_a.dim() == 2, "Inputd must be a 2D matrix."
	assert self.margin > 0.0, 'Margin should be positive value.'
        a_p_pair = Variable(torch.FloatTensor(K, camera_a.size(1)).zero_()).cuda()
        p_a_pair = Variable(torch.FloatTensor(K, camera_a.size(1)).zero_()).cuda()
        a_n_pair = Variable(torch.FloatTensor((P-1)*K, camera_a.size(1)).zero_()).cuda()
        n_a_pair = Variable(torch.FloatTensor((P-1)*K, camera_a.size(1)).zero_()).cuda()
        dist_ap_max, dist_an_min = [], []
	# for each image of each person, find the hardest positive and hardest negative
        for k0 in range(K):    
            for p0 in range(P):
                for k1 in range(K):
                    a_p_pair[k1] = camera_a[P*k0 + p0]
                    p_a_pair[k1] = camera_b[P*k1 + p0]
                dist_ap_max.append(torch.max(pairwise_distance(a_p_pair, p_a_pair)))

                range_no_p0 = range_except_k(p0, P)
                t = 0
                for p1 in range_no_p0:
                    for k2 in range(K):
                        a_n_pair[t] = camera_a[P*k0 + p0]
                        n_a_pair[t] = camera_b[P*k2 + p1]
                        t += 1
                dist_an_min.append(torch.min(pairwise_distance(a_n_pair, n_a_pair)))

        dist_ap_max = torch.cat(dist_ap_max)
        dist_an_min = torch.cat(dist_an_min)

        # dist_hinge = torch.clamp(self.margin + dist_ap_max - dist_an_min, min=0.0)
        dist_soft = torch.log(1 + torch.exp(self.margin + dist_ap_max - dist_an_min))
        # loss = torch.mean(dist_hinge)
        loss = torch.mean(dist_soft)
        return loss


def pairwise_distance(x1, x2, p=2, eps=1e-6):

    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p).sum(dim=1)
    return torch.pow(out, 1. / p)

def non_squared_pairwise_distance(x1, x2):
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.sum(diff, dim=1)
    return out

def range_except_k(n, end, start = 0):
    return range(start, n) + range(n+1, end)

