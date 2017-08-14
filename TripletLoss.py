from __future__ import print_function
import argparse
import numpy as np
import torch

import torch.nn as nn
from torch.autograd import Variable


def range_except_k(n, end, start = 0):
    return range(start, n) + range(n+1, end)

def batch_hard_triplet_margin_loss(camera_a, camera_b, P, K, margin=2.0):
    #camera_a with size (num_person P * num_same K)*num_class
    #camera_b with size (num_person P * num_same K)*num_class
    assert camera_a.size() == camera_b.size(), "Input sizes between camera_a and camera_b must be equal."
    assert camera_a.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    a_p_pair = Variable(torch.FloatTensor(K, camera_a.size(1)).zero_()).cuda()
    p_a_pair = Variable(torch.FloatTensor(K, camera_a.size(1)).zero_()).cuda()
    a_n_pair = Variable(torch.FloatTensor((P-1)*K, camera_a.size(1)).zero_()).cuda()
    n_a_pair = Variable(torch.FloatTensor((P-1)*K, camera_a.size(1)).zero_()).cuda()
    dist_ap_max, dist_an_min = [], []

    for k0 in range(K):    
        for p0 in range(P):
            for i in range(K):
                a_p_pair[i] = camera_a[P*k0 + p0]
                p_a_pair[i] = camera_b[P*i + p0]
            dist_ap_max.append(torch.max(pairwise_distance(a_p_pair, p_a_pair)))

            range_no_n = range_except_k(p0, P)
            t = 0
            for j in range_no_n:
                for k2 in range(K):
                    a_n_pair[t] = camera_a[P*k0 + p0]
                    n_a_pair[t] = camera_b[P*k2 + j]
                    t += 1
            dist_an_min.append(torch.min(pairwise_distance(a_n_pair, n_a_pair)))

    dist_ap_max = torch.cat(dist_ap_max)
    dist_an_min = torch.cat(dist_an_min)

    dist_hinge = torch.clamp(margin + dist_ap_max - dist_an_min, min=0.0)
    # dist_soft = torch.log(1 + torch.exp(margin + dist_ap_max - dist_an_min))
    loss = torch.mean(dist_hinge)
    # loss = torch.mean(dist_soft)
    return loss

def batch_all_triplet_margin_loss(camera_a, camera_b, P, K, margin=1.0):
    #camera_a with size (num_person P * num_same K)*num_class
    #camera_b with size (num_person P * num_same K)*num_class
    assert camera_a.size() == camera_b.size(), "Input sizes between camera_a and camera_b must be equal."
    assert camera_a.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    anchor = Variable(torch.FloatTensor(P*K*K*(P-1)*K, camera_a.size(1)).zero_()).cuda()
    positive = Variable(torch.FloatTensor(P*K*K*(P-1)*K, camera_a.size(1)).zero_()).cuda()
    negative = Variable(torch.FloatTensor(P*K*K*(P-1)*K, camera_a.size(1)).zero_()).cuda()
    i = 0
    for k0 in range(K):
        for p0 in range(P):
            range_no_p0 = range_except_k(p0, P)
            for k1 in range(K):
                for p1 in range_no_p0:
                    for k2 in range(K):
                        anchor[i] = camera_a[P*k0+p0]
                        positive[i] = camera_b[P*k1+p0]
                        negative[i] = camera_b[P*k2+p1]
                        i += 1

    d_p = pairwise_distance(anchor, positive)
    d_n = pairwise_distance(anchor, negative)
    # dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)
    dist_soft = torch.log(1 + torch.exp(margin + d_p - d_n))
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
