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

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from cuhk03_alexnet import AlexNet
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
parser.add_argument('--train-batch-size', type=int, default=24, metavar='N',
                    help='input batch size for training (default: 24)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
# lr=0.1 for resnet, 0.01 for alexnet and vgg
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.005, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='cuhk03_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: cuhk03_checkpoint)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# get triplet dataset
def _get_triplet_data(train):
    with h5py.File('cuhk-03.h5','r') as ff:
        class_num = 800
        a = np.array([ff['a']['train'][str(i)][0] for i in range(class_num)])
        a_label = np.array([i for i in range(class_num)])
        b = np.array([ff['b'][train][str(i)][0] for i in range(class_num)])
        b_label = np.array([i for i in range(class_num)])
        c = np.array([ff['a'][train][str(i)][0] for i in reversed(range(class_num))])
        c_label = np.array([i for i in reversed(range(class_num))])
    	a_trans = a.transpose(0, 3, 1, 2)
    	b_trans = b.transpose(0, 3, 1, 2)
        c_trans = c.transpose(0, 3, 1, 2)
        dataset = []
        label = []
        for i in range(class_num):
            dataset.append(a_trans[i])
            dataset.append(b_trans[i])
            dataset.append(c_trans[i])
            label.append(a_label[i])
            label.append(b_label[i])
            label.append(c_label[i])

    	dataset = np.array(dataset)
	label = np.array(label)
	triplet_dataset = torch.from_numpy(dataset)
    	triplet_label = torch.from_numpy(label)

        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])
        for j in range(class_num*3):
            triplet_dataset[j] = transform(triplet_dataset[j])

        return triplet_dataset, triplet_label, class_num


# get validation dataset of five camera pairs
def _get_data(val_or_test):
    with h5py.File('cuhk-03.h5','r') as ff:
    	num = 100
    	num1 = 100
    	num2 = 100
    	a = np.array([ff['a'][val_or_test][str(i)][1] for i in range(num1)])
    	b = np.array([ff['b'][val_or_test][str(i)][1] for i in range(num2)])
    	a_trans = a.transpose(0, 3, 1, 2)
    	b_trans = b.transpose(0, 3, 1, 2)
    	camere1 = torch.from_numpy(a_trans)
    	camere2 = torch.from_numpy(b_trans)
        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])
        # for j in range(num):
            # camere1[j] = transform(camere1[j])
            # camere2[j] = transform(camere2[j])
        for j in range(num1):
            camere1[j] = transform(camere1[j])
        for j in range(num2):
            camere2[j] = transform(camere2[j])

        return camere1, camere2



def train_model(train_loader, model, criterion, optimizer, epoch):

    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        pair_num = args.train_batch_size / 3
	# print(inputs.size())
	# print(pair_num)
        anchor = torch.FloatTensor(pair_num, inputs.size(1), inputs.size(2), inputs.size(3)).zero_()
        positive = torch.FloatTensor(pair_num, inputs.size(1), inputs.size(2), inputs.size(3)).zero_()
        negative = torch.FloatTensor(pair_num, inputs.size(1), inputs.size(2), inputs.size(3)).zero_()
        for i in range(pair_num):
            anchor[i] = inputs[3*i]
            positive[i] = inputs[3*i+1]
            negative[i] = inputs[3*i+2]

        anchor = anchor.float()  # with size of (batch_size/3 * 3 * 224 * 224)
        positive = positive.float()
        negative = negative.float()
        if args.cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)
        optimizer.zero_grad()
        # compute output
        outputs1 = model(anchor)
        outputs2 = model(positive)
        outputs3 = model(negative)
        loss = criterion(outputs1, outputs2, outputs3)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            print()



def cmc(model, val_or_test='test'):

    a,b = _get_data(val_or_test)
    # camera1 as probe, camera2 as gallery
    def _cmc_curve(model, camera1, camera2, rank_max=20):
        # num = 100  # 100
        num1 = 100  # camera1
        num2 = 100  # camera2
        rank = []
        score = []
        # camera_batch1 = camera1
        camera_batch1 = camera2
        camera1 = camera1.float()
        camera_batch1 = camera_batch1.float()
        camera2 = camera2.float()
        if args.cuda:
            camera1, camera_batch1, camera2 = camera1.cuda(), camera_batch1.cuda(), camera2.cuda()
        camera1, camera_batch1, camera2 = Variable(camera1), Variable(camera_batch1), Variable(camera2)
        feature2_batch = model(camera2)       # with size 100 * 4096
        for i in range(num1):
            for j in range(num2):
                camera_batch1[j] = camera1[i]
            feature1_batch = model(camera_batch1) # with size 100 * 4096
            pdist = nn.PairwiseDistance(2)
            dist_batch = pdist(feature1_batch, feature2_batch)  # with size 100 * 1
            dist_np = dist_batch.cpu().data.numpy()
    	    # dist_np = np.reshape(dist_np, (num))
    	    dist_np = np.reshape(dist_np, (num2))
            dist_sorted = np.argsort(dist_np)
    	    if i < 30:
		print(dist_sorted[:10])
            # for k in range(num):
            for k in range(num2):
                if dist_sorted[k] == i:
                    rank.append(k+1)
                    break
        rank_val = 0
        for i in range(rank_max):
            rank_val = rank_val + len([j for j in rank if i == j-1])
            # score.append(rank_val / float(num))
            score.append(rank_val / float(num1))
        return np.array(score)
    return _cmc_curve(model,a,b)


def main():

    triplet_dataset, triplet_label, class_num = _get_triplet_data('train')
    print('train data size', triplet_dataset.size())
    print('train target size', triplet_label.size())
    train_data = data_utils.TensorDataset(triplet_dataset, triplet_label)
    train_loader = data_utils.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if 0:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)
        model = torch.nn.DataParallel(model)

    if 1:
        model = models.alexnet(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, class_num)
        model.features = torch.nn.DataParallel(model.features)

    if args.cuda:
        model.cuda()
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    if args.cuda:
        criterion = nn.TripletMarginLoss(margin=1.0, p=2).cuda()

    title = 'CUHK03-AlexNet'
    date_time = get_datetime()
    # Train
    for epoch in range(1, args.epochs + 1):
        lr, optimizer = exp_lr_scheduler(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, lr))
        print()
        train_model(train_loader, model, criterion, optimizer, epoch)

    # Test
    model.eval()
    if 0:
        model.fc = ''
        # model = torch.nn.DataParallel(model)
    	torch.save(model, 'triplet_resnet_trained.pth')
    if 1:
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        # model.features = torch.nn.DataParallel(model.features)
    	torch.save(model, 'triplet_alexnet_trained.pth')


    score_array = cmc(model)
    print(score_array)

    print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}'.format(
        score_array[0], score_array[4]))

def use_trained_model():

    if 0:
    	model = torch.load('triplet_resnet_trained.pth')
    if 1:
    	model = torch.load('triplet_alexnet_trained.pth')

    score_array = cmc(model)
    print(score_array)

    print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}'.format(
        score_array[0], score_array[4]))


def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=20):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
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
