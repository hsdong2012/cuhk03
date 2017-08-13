from __future__ import print_function
import argparse
import h5py
import sys
import os
import time
import datetime
import shutil
import random
import numpy as np
import torch
import scipy.misc
import scipy.io as sio
import itertools as it
from torchvision import datasets, transforms
from torchvision import models
import torch.utils.data as data_utils

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from function import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 160)')
parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 60)')
# lr=0.1 for resnet, 0.01 for alexnet and vgg
parser.add_argument('--lr', type=float, default=0.0004, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.005, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='variant_triplet_main_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: cuhk03_checkpoint)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True


# get triplet dataset
def range_except_k(n, end, start = 0):
    return range(start, n) + range(n+1, end)


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
        # print(a_tensor.size())

        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])
        for j in range(class_num):
            for k in range(3):
                a_tensor[j][k] = transform(a_tensor[j][k])
                b_tensor[j][k] = transform(b_tensor[j][k])
	
	
	triplet_dataset = torch.stack((a_tensor, b_tensor), 1) #class_num*2*3*(3*224*224
	triplet_label = id_tensor
        return triplet_dataset, triplet_label
	
	

	'''	
        # pair_num = 200*199
        pair_num = 40*class_num
        # pair_num = 20000

        triplet_temp = torch.FloatTensor(pair_num, 3, a_tensor.size(2), a_tensor.size(3), a_tensor.size(4)).zero_()
        triplet_id_temp = torch.LongTensor(pair_num, 3).zero_()
	
	"""	
        i = 0
        for k in range(200):
            range_no_k = range_except_k(k, 200)
            for k1 in range_no_k:
                j0 = random.randint(0, 2)
                j1 = random.randint(0, 2)
                j2 = random.randint(0, 2)
                triplet_temp[i][0] = a_tensor[k][0]
                triplet_id_temp[i][0] = k
                triplet_temp[i][1] = b_tensor[k][0]
                triplet_id_temp[i][1] = k
                triplet_temp[i][2] = b_tensor[k1][0]
                triplet_id_temp[i][2] = k1
                i += 1
	"""
	
	
        i = 0
        for j in range(40):
            for k in range(class_num):
                range_no_k = range_except_k(k, class_num)
                k1 = random.choice(range_no_k)
                j0 = random.randint(0, 2)
                j1 = random.randint(0, 2)
                j2 = random.randint(0, 2)
                triplet_temp[i][0] = a_tensor[k][j0]
                triplet_id_temp[i][0] = k
                triplet_temp[i][1] = b_tensor[k][j1]
                triplet_id_temp[i][1] = k
                triplet_temp[i][2] = b_tensor[k1][j2]
                triplet_id_temp[i][2] = k1
                i += 1
	

        triplet_dataset = triplet_temp
        triplet_label = triplet_id_temp
	
        return triplet_dataset, triplet_label
	'''

# get validation dataset of five camera pairs
def _get_data(val_or_test):
    with h5py.File('cuhk-03.h5','r') as ff:
    	num1 = 80  # camera1, probe
        num2 = 80  # camera2, gallery, 100 >= num2 >= num1
    	a = np.array([ff['a'][val_or_test][str(i)][0] for i in range(num1)])
    	b = np.array([ff['b'][val_or_test][str(i)][0] for i in range(num2)])
    	a_trans = a.transpose(0, 3, 1, 2)
    	b_trans = b.transpose(0, 3, 1, 2)
    	camere1 = torch.from_numpy(a_trans)
    	camere2 = torch.from_numpy(b_trans)
        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])

        for j in range(num1):
            camere1[j] = transform(camere1[j])
        for j in range(num2):
            camere2[j] = transform(camere2[j])

        return (camere1, camere2)


def train_model(train_loader, model, optimizer, epoch):

    model.train()
    losses = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
	
	size0 = inputs.size(0)
        camera_pair = torch.split(inputs, 1, 1)
        camera_a_3 = torch.squeeze(camera_pair[0]) # batch_size *3 * (3*224*224)
        camera_b_3 = torch.squeeze(camera_pair[1])
	camera_a_pair = torch.split(camera_a_3, 1, 1)
	camera_b_pair = torch.split(camera_b_3, 1, 1)
	camera_a = torch.squeeze(camera_a_pair[0]) # batch_size * (3*224*224)
	camera_b = torch.squeeze(camera_b_pair[0])
	camera_a,camera_b = Variable(camera_a).float(), Variable(camera_b).float()
	if args.cuda:
	    camera_a, camera_b = camera_a.cuda(), camera_b.cuda()
	outputs_a = model(camera_a)
	outputs_b = model(camera_b)
	actual_size = size0*(size0-1)
        outputs1 = torch.FloatTensor(actual_size, outputs_a.size(1)).zero_()
        outputs2 = torch.FloatTensor(actual_size, outputs_a.size(1)).zero_()
        outputs3 = torch.FloatTensor(actual_size, outputs_a.size(1)).zero_()
	outputs1, outputs2, outputs3 = Variable(outputs1).cuda(), Variable(outputs2).cuda(), Variable(outputs3).cuda()
	k = 0
	for p0 in range(size0):
	    range_no_p0 = range_except_k(p0, size0)
	    for p1 in range_no_p0:
		outputs1[k] = outputs_a[p0]
		outputs2[k] = outputs_b[p0]
		outputs3[k] = outputs_b[p1]
		k += 1
		


	"""
	size0 = inputs.size(0)
        camera_pair = torch.split(inputs, 1, 1)
        camera_a = torch.squeeze(camera_pair[0]) # batch_size * 3 *(3*224*224)
        camera_b = torch.squeeze(camera_pair[1])
	actual_size = size0*2*2*(size0-1)*2
        anchor_temp = torch.FloatTensor(actual_size, camera_a.size(2), camera_a.size(3), camera_a.size(4)).zero_()
        positive_temp = torch.FloatTensor(actual_size, camera_a.size(2), camera_a.size(3), camera_a.size(4)).zero_()
        negative_temp = torch.FloatTensor(actual_size, camera_a.size(2), camera_a.size(3), camera_a.size(4)).zero_()
	k = 0
	for p0 in range(size0):
	    range_no_p0 = range_except_k(p0, size0)
	    for i0 in range(2):
		for i1 in range(2):
		    for p1 in range_no_p0:
			for i2 in range(2):
			    anchor_temp[k] = camera_a[p0][i0]
			    positive_temp[k] = camera_b[p0][i1]
			    negative_temp[k] = camera_b[p1][i2]
			    k += 1
	anchor = anchor_temp
	positive = positive_temp
	negative = negative_temp
	if 0:		
	    a_np = anchor.numpy()		
	    p_np = positive.numpy()		
	    n_np = negative.numpy()		
	    a_img = a_np.transpose(0, 2, 3, 1)		
	    p_img = p_np.transpose(0, 2, 3, 1)		
	    n_img = n_np.transpose(0, 2, 3, 1)
	    file_path = 'batch_all_normalize_image/'		
	    directory = os.path.dirname(file_path)		
	    if not os.path.exists(directory):		
	        os.makedirs(directory)		
	    for k in it.chain(xrange(40,41), xrange(60, 61), xrange(80,81)):
		scipy.misc.imsave(directory+'/anchor'+'_'+str(k)+'.png', a_img[k])
		scipy.misc.imsave(directory+'/positive'+'_'+str(k)+'.png', p_img[k])
		scipy.misc.imsave(directory+'/negative'+'_'+str(k)+'.png', n_img[k])
	    sys.exit('exit')
	"""

	'''	
        triplet_pair = torch.split(inputs, 1, 1)
        anchor = torch.squeeze(triplet_pair[0])
        positive = torch.squeeze(triplet_pair[1])
        negative = torch.squeeze(triplet_pair[2])
		

        inputs_cat = Variable(torch.cat((anchor, positive, negative), 0)).cuda()
        outputs_cat = model(inputs_cat)
        outputs_split = torch.split(outputs_cat, anchor.size(0), 0)
        outputs1 = outputs_split[0]
        outputs2 = outputs_split[1]
        outputs3 = outputs_split[2]
        '''

        outputs1, outputs2, outputs3 = torch.squeeze(outputs1), torch.squeeze(outputs2), torch.squeeze(outputs3)

        # compute loss
        criterion = nn.TripletMarginLoss(margin=4.0, p=2)
        if args.cuda:
            criterion = nn.TripletMarginLoss(margin=4.0, p=2).cuda()
	loss = criterion(outputs1, outputs2, outputs3) 
        # loss = variant_triplet_margin_loss(outputs1, outputs2, outputs3, margin=3.0)
        losses.update(loss.data[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            print()

    return losses.avg


def cmc(model, val_or_test='test'):

    model.eval()
    a,b = _get_data(val_or_test)
    # camera1 as probe, camera2 as gallery
    def _cmc_curve(model, camera1, camera2, rank_max=20):
        num1 = 80  # camera1, probe
        num2 = 80  # camera2, gallery, 100 >= num2 >= num1
        rank = []
        score = []
        camera_batch1 = camera2
        camera1 = camera1.float()
        camera_batch1 = camera_batch1.float()
        camera2 = camera2.float()
        if args.cuda:
            camera1, camera_batch1, camera2 = camera1.cuda(), camera_batch1.cuda(), camera2.cuda()
        camera1, camera_batch1, camera2 = Variable(camera1), Variable(camera_batch1), Variable(camera2)
        feature2_batch = model(camera2)       # with size 100 * 4096
        feature2_batch = torch.squeeze(feature2_batch)

        for i in range(num1):
            for j in range(num2):
                camera_batch1[j] = camera1[i]
            feature1_batch = model(camera_batch1) # with size 100 * 4096
            feature1_batch = torch.squeeze(feature1_batch)

            pdist = nn.PairwiseDistance(2)
            dist_batch = pdist(feature1_batch, feature2_batch)  # with size 100 * 1
            # dist_batch = variant_pairwise_distance(feature1_batch, feature2_batch)  # with size 100 * 1
            distance = torch.squeeze(dist_batch)
            dist_value, dist_indices = torch.sort(distance)
            dist_indices = dist_indices.data.cpu().numpy()

            if i < 30:
                print(dist_indices[:10])
            for k in range(num2):
                if dist_indices[k] == i:
                    rank.append(k+1)
                    break

        rank_val = 0
        for i in range(rank_max):
            rank_val = rank_val + len([j for j in rank if i == j-1])
            score.append(rank_val / float(num1))

        score_array = np.array(score)
        print(score_array)
        print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}\t''Top10(accuracy) : {:.3f}'.format(score_array[0], score_array[4], score_array[9]))
        return score_array

    return _cmc_curve(model,a,b)



def main():

    model_name = 'resnet50'
    original_model = models.resnet50(pretrained=True)
    num_ftrs = original_model.fc.in_features
    # new_model = nn.Sequential(*list(original_model.children())[:-1])

    new_model = original_model
    new_model.fc = nn.Linear(num_ftrs, 1024)
    new_model.add_module('bn0', nn.BatchNorm1d(1024))
    new_model.add_module('relu0', nn.ReLU(inplace=True))
    new_model.add_module('fc2', nn.Linear(1024, 128))
    import torch.nn.init as init
    init.kaiming_normal(new_model.fc.weight, mode='fan_out')
    init.constant(new_model.fc.bias, 0)
    init.constant(new_model.bn0.weight, 1)
    init.constant(new_model.bn0.bias, 0)
    init.normal(new_model.fc2.weight, std=0.001)
    init.constant(new_model.fc2.bias, 0)


    new_model = torch.nn.DataParallel(new_model)
    if args.cuda:
        new_model.cuda()
    print(new_model)


    triplet_dataset, triplet_label = _get_triplet_data()
    print('train data  size: ', triplet_dataset.size())
    print('train target size', triplet_label.size())
    train_data = data_utils.TensorDataset(triplet_dataset, triplet_label)
    train_loader = data_utils.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

    optimizer = optim.Adam(new_model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(new_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.TripletMarginLoss(margin=4.0, p=2)
    if args.cuda:
        criterion = nn.TripletMarginLoss(margin=4.0, p=2).cuda()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    title = 'CUHK03-Dataset'
    date_time = get_datetime()
    log_filename = 'log-triplet-'+str(triplet_dataset.size(0))+'-'+model_name+'-'+date_time+'.txt'
    logger = Logger(os.path.join(args.checkpoint, log_filename), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Test Top1', 'Test Top5', 'Test Top10'])

    # Train
    for epoch in range(1, args.epochs + 1):
        lr, optimizer = exp_lr_scheduler(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, lr))
        print()
        loss = train_model(train_loader, new_model, optimizer, epoch)
    	score_array = cmc(new_model)
        logger.append([lr, loss, score_array[0], score_array[4], score_array[9]])

    logger.close()

    # save model
    # torch.save(new_model, 'variant_triplet_resnet50_trained.pth')


def use_trained_model():

    model = torch.load('variant_triplet_resnet50_trained.pth')

    score_array = cmc(model)
    print(score_array)
    print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}'.format(
        score_array[0], score_array[4]))


def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))
    # lr = init_lr * (0.2**(epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, optimizer


def get_datetime():
    now_datetime = str(datetime.datetime.now())
    array0 = now_datetime.split(' ')
    yymmdd = array0[0]
    time_array = array0[1].split(':')
    hour_min = time_array[0]+time_array[1]
    date_time = yymmdd+'-'+hour_min
    return date_time

if __name__ == '__main__':
    main()
    # use_trained_model()
