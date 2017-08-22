from __future__ import print_function
import argparse
import h5py
import sys
import os
import os.path as osp
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
import torch.utils.data as data_utils

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import models
from TripletLoss import batch_hard_triplet_margin_loss, batch_all_triplet_margin_loss
from dataset import _get_triplet_data, _get_data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
# 64 for batch_hard, 4 for batch_all(4 GPU)
parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 160)')
parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 60)')
# lr=0.1 for SGD, 0.0003 for Adam
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
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
parser.add_argument('--logs-dir',default='log_resnet_new', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# model
parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    choices=models.names())
parser.add_argument('--features', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

if not os.path.isdir(args.logs_dir):
    mkdir_p(args.logs_dir)

def range_except_k(n, end, start = 0):
    return range(start, n) + range(n+1, end)


def split_cameras(inputs):
    # inputs: batch_size * 2(cameras) * k(images of same person) * (3*224*224)
    camera_pair = torch.split(inputs, 1, 1)
    camera_a_k = torch.squeeze(camera_pair[0]) # batch_size *k * (3*224*224)
    camera_b_k = torch.squeeze(camera_pair[1])
    camera_a_pair = torch.split(camera_a_k, 1, 1)
    camera_b_pair = torch.split(camera_b_k, 1, 1)
    camera_a = torch.squeeze(torch.cat(camera_a_pair, 0))
    camera_b = torch.squeeze(torch.cat(camera_b_pair, 0))
    # camera_a: (batch_size*k) * (3*224*224)
    return camera_a, camera_b


def train_model(train_loader, model, optimizer, epoch):

    model.train()
    losses = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

	num_person = inputs.size(0)
	num_same = inputs.size(2)

	camera_a, camera_b = split_cameras(inputs)

	camera_a,camera_b = Variable(camera_a).float(), Variable(camera_b).float()
	if args.cuda:
	    camera_a, camera_b = camera_a.cuda(), camera_b.cuda()

	outputs_a = model(camera_a)
	outputs_b = model(camera_b)
	outputs_a, outputs_b = torch.squeeze(outputs_a), torch.squeeze(outputs_b)
	
	# loss = batch_all_triplet_margin_loss(outputs_a, outputs_b, num_person, num_same)
	loss = batch_hard_triplet_margin_loss(outputs_a, outputs_b, num_person, num_same)
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
    a,b = _get_data(val_or_test, num1=100, num2=100)
    # camera1 as probe, camera2 as gallery
    def _cmc_curve(model, camera1, camera2, rank_max=20):
        num1 = camera1.size(0)  # camera1, probe
        num2 = camera2.size(0)  # camera2, gallery, 100 >= num2 >= num1
        rank = []
        score = []
        camera_batch1 = camera2
        camera1 = camera1.float()
        camera_batch1 = camera_batch1.float()
        camera2 = camera2.float()
        if args.cuda:
            camera1, camera_batch1, camera2 = camera1.cuda(), camera_batch1.cuda(), camera2.cuda()
        camera1, camera_batch1, camera2 = Variable(camera1), Variable(camera_batch1), Variable(camera2)
        feature2_batch = model(camera2)       # num2 * num_features
        feature2_batch = torch.squeeze(feature2_batch)

        for i in range(num1):
            for j in range(num2):
                camera_batch1[j] = camera1[i]
            feature1_batch = model(camera_batch1) # num1 * num_features
            feature1_batch = torch.squeeze(feature1_batch)

            pdist = nn.PairwiseDistance(2)
            dist_batch = pdist(feature1_batch, feature2_batch) 
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

    model_name = 'resnet50_new'
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    model = torch.nn.DataParallel(model)
    if args.cuda:
        model.cuda()

    triplet_dataset, triplet_label = _get_triplet_data()
    print('train data  size: ', triplet_dataset.size())
    print('train target size', triplet_label.size())
    train_data = data_utils.TensorDataset(triplet_dataset, triplet_label)
    train_loader = data_utils.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(new_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    title = 'CUHK03-Dataset'
    date_time = get_datetime()
    triplet_batch = ['bh', 'ba']
    loss_margin = ['hm', 'sm']
    log_filename = 'log-triplet-'+triplet_batch[0]+'-'+loss_margin[0]+'-'+model_name+'-'+date_time+'.txt'
    logger = Logger(os.path.join(args.logs_dir, log_filename), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Test Top1', 'Test Top5', 'Test Top10'])

    # Train
    best_acc = 0  # best test accuracy
    for epoch in range(1, args.epochs + 1):
        lr, optimizer = exp_lr_scheduler(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, lr))
        print()
        loss = train_model(train_loader, model, optimizer, epoch)
    	score_array = cmc(model)
        logger.append([lr, loss, score_array[0], score_array[4], score_array[9]])
	# save model
	test_acc = score_array[0]
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
	save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch,
            'best_top1': best_acc,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))	

    logger.close()
    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    score_array = cmc(model)


def use_trained_model():

    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)
    model = torch.nn.DataParallel(model)
    if args.cuda:
        model.cuda()

    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    score_array = cmc(model)


def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=15):
    """Decay learning rate by a factor of 0.5/0.2 every lr_decay_epoch epochs."""
    if epoch < 30:
    	lr = init_lr * (0.5**(epoch // lr_decay_epoch))
    else:
	lr = init_lr * 0.5
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
    # main()
    use_trained_model()
