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

def variant_pairwise_distance(x1, x2):
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.sum(dim=1, keepdim=True)
    return out


def variant_triplet_margin_loss(anchor, positive, negative, margin=1.0, swap=False):

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = variant_pairwise_distance(anchor, positive, p, eps)
    d_n = variant_pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = variant_pairwise_distance(positive, negative)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.log(1 + torch.exp(margin + d_p - d_n))
    loss = torch.mean(dist_hinge)
    return loss
