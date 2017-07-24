from __future__ import print_function
import argparse
import h5py
import sys
import os
import time
import datetime
import shutil
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import models
import torch.utils.data as data_utils
import random


def range_except_k(n, end, start = 0):
    return range(start, n) + range(n+1, end)

def _get_triplet_data():
    with h5py.File('triplet-cuhk-03.h5', r) as ff:
        class_num = len(ff['a']['train'].keys())
        temp_a = []
        temp_b = []
        temp_id = []
        for i in range(class_num):
            for k in range(5):
                temp_id.append(i)
                temp_a.append(np.array(ff['a']['train'][str(i)][k]))
                temp_b.append(np.array(ff['b']['train'][str(i)][k]))

        imageset_a = np.array(temp_a)
        imageset_b = np.array(temp_b)
        a_trans = imageset_a.transpose(0, 3, 1, 2)
        b_trans = imageset_b.transpose(0, 3, 1, 2)
        a_tensor = torch.from_numpy(a_trans)
        b_tensor = torch.from_numpy(b_trans)

        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])
        for j in range(class_num*5):
            a_tensor[j] = transform(a_tensor[j])
            b_tensor[j] = transform(b_tensor[j])

        camera1_dataset = a_tensor.resize_(class_num, 5, a_tensor.size(1), a_tensor.size(2), a_tensor.size(3))
        camera2_dataset = b_tensor.resize_(class_num, 5, b_tensor.size(1), b_tensor.size(2), b_tensor.size(3))

        pair_num = 10000
        triplet_temp = torch.FloatTensor(pair_num, 3, a_tensor.size(1), a_tensor.size(2), a_tensor.size(3)).zero_()
        triplet_id_temp = torch.LongTensor(pair_num, 3)
        for i in range(pair_num):
            k = random.randint(0, class_num)
            k1 = range_except_k(k, class_num)
            j0 = random.randint(0, 5)
            j1 = random.randint(0, 5)
            j2 = random.randint(0, 5)
            triplet_temp[i][0] = camera1_dataset[k][j0]
            triplet_id_temp[i][0] = k
            triplet_temp[i][1] = camera2_dataset[k][j1]
            triplet_id_temp[i][1] = k
            triplet_temp[i][2] = camera1_dataset[k1][j2]
            triplet_id_temp[i][2] = k1

        triplet_dataset = triplet_temp
        triplet_label = triplet_id_temp

        return triplet_dataset, triplet_label

if 1:
    triplet_dataset, triplet_label= _get_triplet_data()
    print('train data  size: ', triplet_dataset.size())
    print('train target size', triplet_label.size())
