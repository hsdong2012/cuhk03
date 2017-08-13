from __future__ import print_function
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import models
import torch.utils.data as data_utils

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

def variant_pairwise_distance(x1, x2, p=1, eps=1e-6):
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.sum(diff, dim=1)
    return out


def variant_triplet_margin_loss(anchor, positive, negative, margin=4.0, swap=False):

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    # d_p = variant_pairwise_distance(anchor, positive)
    # d_n = variant_pairwise_distance(anchor, negative)
    d_p = pairwise_distance(anchor, positive)
    d_n = pairwise_distance(anchor, negative)
    if swap:
        # d_s = variant_pairwise_distance(anchor, negative)
        d_s = pairwise_distance(positive, negative)
        d_n = torch.min(d_n, d_s)

    dist_soft = torch.log(1 + torch.exp(margin + d_p - d_n))
    loss = torch.mean(dist_soft)
    return loss

def pairwise_distance(x1, x2, p=2, eps=1e-6):
    
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p).sum(dim=1)
    return torch.pow(out, 1. / p)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False):

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n = pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)
    loss = torch.mean(dist_hinge)
    """dist_hinge_np = dist_hinge.cpu().data
    nonzero_idx = torch.nonzero(dist_hinge_np)
    nonzero_num = nonzero_idx.size(0)
    total_loss = torch.sum(dist_hinge)
    loss = total_loss/nonzero_num"""
    return loss
