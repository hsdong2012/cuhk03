from __future__ import print_function
import h5py
import sys
import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms

import torch.nn as nn
from reid.utils.data import transforms as T

# get triplet dataset
def _get_triplet_data():
    with h5py.File('cuhk-03.h5', 'r') as ff:
        class_num = len(ff['a']['train'].keys())
        temp_a = []
        temp_b = []
        a_temp = []
        b_temp = []
        temp_id = []
        for i in range(class_num):
            len1 = len(ff['a']['train'][str(i)])
            len2 = len(ff['b']['train'][str(i)])
            if len1 < 3 or len2 < 3:
                class_num -= 1
            if len1 >= 3 and len2 >= 3:
                for k in range(3):
                    temp_a.append(np.array(ff['a']['train'][str(i)][k]))
                    temp_b.append(np.array(ff['b']['train'][str(i)][k]))
                a_temp.append(temp_a)
                b_temp.append(temp_b)
                temp_id.append(i)
                temp_a = []
                temp_b = []

	person_id = np.array(temp_id)
        camera_a = np.array(a_temp)
        camera_b = np.array(b_temp)
        a_trans = camera_a.transpose(0, 1, 4, 2, 3)
        b_trans = camera_b.transpose(0, 1, 4, 2, 3)
	id_tensor = torch.from_numpy(person_id)
        a_tensor = torch.from_numpy(a_trans)
        b_tensor = torch.from_numpy(b_trans)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])
        for j in range(class_num):
            for k in range(3):
                a_tensor[j][k] = normalize(a_tensor[j][k])
                b_tensor[j][k] = normalize(b_tensor[j][k])
	
	
	triplet_dataset = torch.stack((a_tensor, b_tensor), 1) #class_num*2*3*(3*224*224)
	triplet_label = id_tensor

        return triplet_dataset, triplet_label
	
	

def _get_data(val_or_test, num1, num2):
    with h5py.File('cuhk-03.h5','r') as ff:
    	# num1 for camera1, probe
        # num2 for camera2, gallery, 100 >= num2 >= num1
    	a = np.array([ff['a'][val_or_test][str(i)][0] for i in range(num1)])
    	b = np.array([ff['b'][val_or_test][str(i)][0] for i in range(num2)])
    	a_trans = a.transpose(0, 3, 1, 2)
    	b_trans = b.transpose(0, 3, 1, 2)
    	camera1 = torch.from_numpy(a_trans)
    	camera2 = torch.from_numpy(b_trans)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])

        for j in range(num1):
            camera1[j] = normalize(camera1[j])
        for j in range(num2):
            camera2[j] = normalize(camera2[j])

        return (camera1, camera2)



def new_get_triplet_data():
    height, width = 224, 224
    with h5py.File('cuhk-03.h5', 'r') as ff:
        class_num = len(ff['a']['train'].keys())
        temp_a = []
        temp_b = []
        a_temp = []
        b_temp = []
        temp_id = []
        for i in range(class_num):
            len1 = len(ff['a']['train'][str(i)])
            len2 = len(ff['b']['train'][str(i)])
            if len1 < 3 or len2 < 3:
                class_num -= 1
            if len1 >= 3 and len2 >= 3:
                for k in range(3):
                    temp_a.append(np.array(ff['a']['train'][str(i)][k]))
                    temp_b.append(np.array(ff['b']['train'][str(i)][k]))
                a_temp.append(temp_a)
                b_temp.append(temp_b)
                temp_id.append(i)
                temp_a = []
                temp_b = []

	person_id = np.array(temp_id)
        camera_a = np.array(a_temp, dtype=np.uint8)
        camera_b = np.array(b_temp, dtype=np.uint8)

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	train_transformer = T.Compose([
	    T.ToPILImage(),
            # T.RandomSizedRectCrop(height, width),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer
	])

	a_tensor = torch.FloatTensor(class_num, 3, 3, height, width).zero_()
	b_tensor = torch.FloatTensor(class_num, 3, 3, height, width).zero_()
	id_tensor = torch.from_numpy(person_id)

        for j in range(class_num):
            for k in range(3):
                a_tensor[j][k] = train_transformer(camera_a[j][k])
                b_tensor[j][k] = train_transformer(camera_b[j][k])
	
	
	triplet_dataset = torch.stack((a_tensor, b_tensor), 1) #class_num*2*3*(3*height*width)
	triplet_label = id_tensor

        return triplet_dataset, triplet_label

def new_get_data(val_or_test):
    
    height, width = 224, 224
    with h5py.File('cuhk-03.h5','r') as ff:
    	num1 = 100  # camera1, probe
        num2 = 100  # camera2, gallery, 100 >= num2 >= num1
    	a = np.array([ff['a'][val_or_test][str(i)][0] for i in range(num1)], dtype=np.uint8)
    	b = np.array([ff['b'][val_or_test][str(i)][0] for i in range(num2)], dtype=np.uint8)
    
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	test_transformer = T.Compose([
	    T.ToPILImage(),
            # T.RectScale(height, width),
            T.ToTensor(),
            normalizer
	])

	camera1 = torch.FloatTensor(num1, 3, height, width).zero_()
	camera2 = torch.FloatTensor(num2, 3, height, width).zero_()
        for j in range(num1):
            camera1[j] = test_transformer(a[j])
        for j in range(num2):
            camera2[j] = test_transformer(b[j])

        return camera1, camera2

