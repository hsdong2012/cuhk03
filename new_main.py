from __future__ import print_function
import argparse
import h5py
import sys
import os
import time
import datetime
import shutil
import numpy as np
import scipy.misc
import scipy.io as sio
from tqdm import tqdm
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
parser.add_argument('--train-batch-size', type=int, default=160, metavar='N',
                    help='input batch size for training (default: 160)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
# lr=0.1 for resnet, 0.01 for alexnet and vgg
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
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

#get train dataset of five camera pairs
def _get_train_data(train, group):
    with h5py.File('triplet-cuhk-03.h5', 'r') as ff:
        class_num = len(ff[group][train].keys())
        print("class number: ", class_num)
        temp = []
        num_of_same_image_array = []
        num_sample_total = 0
        for i in tqdm(range(class_num)):
            num_of_same_image = len(ff[group][train][str(i)])
            num_sample_total += num_of_same_image
            num_of_same_image_array.append(num_of_same_image)
            for k in range(num_of_same_image):
                temp.append(np.array(ff[group][train][str(i)][k]))
        image_set = np.array(temp)

        image_id = []
        for i in range(class_num):
            for k in range(num_of_same_image_array[i]):
                image_id.append(i)
        image_id = np.array(image_id)
        targets = torch.from_numpy(image_id)

        data = image_set.transpose(0, 3, 1, 2)
        features = torch.from_numpy(data)

        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])
        for j in range(num_sample_total):
            features[j] = transform(features[j])

        return features, targets, class_num

# get validation dataset of five camera pairs
def _get_data(val_or_test):
    with h5py.File('triplet-cuhk-03.h5','r') as ff:
	num1 = 80
	num2 = 80
	a = np.array([ff['a'][val_or_test][str(i)][1] for i in range(num1)])
	b = np.array([ff['b'][val_or_test][str(i)][1] for i in range(num2)])
	a_trans = a.transpose(0, 3, 1, 2)
	b_trans = b.transpose(0, 3, 1, 2)
	camere1 = torch.from_numpy(a_trans)
	camere2 = torch.from_numpy(b_trans)
        transform = transforms.Normalize(mean=[0.367, 0.362, 0.357], std=[0.244, 0.247, 0.249])

	
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
        inputs = inputs.float()  # with size of (batch_size * 3 * 224 * 224)
        targets = targets.long() # with size of (batch_size)
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            print()

        if batch_idx % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f}({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f}({top5.avg:.3f})'.format(
                   epoch, batch_idx, len(train_loader), loss=losses,
                   top1=top1, top5=top5))
            print()
    print('Train batch size: %s' %(args.train_batch_size))
    print('Top1(train) : {:.3f}%\t''Top5(train) : {:.3f}%'.format(
        top1.avg, top5.avg))
    print('Train Average Loss: {:.4f}'.format(losses.avg))
    print()
    return losses.avg, top1.avg, top5.avg

def cmc(model, val_or_test='test'):

        model.eval()
        a,b = _get_data(val_or_test)

        # camera1 as probe, camera2 as gallery
        def _cmc_curve(model, camera1, camera2, rank_max=20):
            num1 = 80  # camera1
            num2 = 80  # camera2
            rank = []
            score = []
            camera_batch1 = camera2
            camera1 = camera1.float()
            camera2 = camera2.float()
            camera_batch1 = camera_batch1.float()
	    if 0:
		probe = camera1.numpy()
		probe_img = probe.transpose(0, 2, 3, 1)
		file_path = 'probe_with_its_top5/'
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
		    os.makedirs(directory)
		for k in range(6):
		    scipy.misc.imsave(directory+'/probe'+str(k)+'.png', probe_img[k])

            if args.cuda:
                camera1, camera_batch1, camera2 = camera1.cuda(), camera_batch1.cuda(), camera2.cuda()

            camera1, camera_batch1, camera2 = Variable(camera1), Variable(camera_batch1), Variable(camera2)

            feature2_batch = model(camera2)       # with size 100 * 4096
	    feature2_batch = torch.squeeze(feature2_batch)	    
	    # print(feature2_batch.size())
            for i in range(num1):
                for j in range(num2):
                    camera_batch1[j] = camera1[i]

                feature1_batch = model(camera_batch1) # with size 100 * 4096
		feature1_batch = torch.squeeze(feature1_batch)
		
                pdist = nn.PairwiseDistance(2)
                dist_batch = pdist(feature1_batch, feature2_batch)  # with size 100 * 1

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
            return np.array(score)

        return _cmc_curve(model,a,b)


def main():

    camera1_train_features, camera1_train_targets, class_num = _get_train_data('train', 'a')
    camera2_train_features, camera2_train_targets, class_num = _get_train_data('train', 'b')
    train_features = torch.cat((camera1_train_features, camera2_train_features), 0)
    train_targets = torch.cat((camera1_train_targets, camera2_train_targets), 0)
    print('train data size', train_features.size())
    print('train target size', train_targets.size())
    train = data_utils.TensorDataset(train_features, train_targets)
    train_loader = data_utils.DataLoader(train, batch_size=args.train_batch_size, shuffle=True)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if 1:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)
        model = torch.nn.DataParallel(model)
	# print(model)
    if 0:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)
        model = torch.nn.DataParallel(model)
	# print(model)
    if 0:
        model = models.alexnet(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, class_num)
        model.features = torch.nn.DataParallel(model.features)
       

    if args.cuda:
        model.cuda()
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()

    title = 'CUHK03-AlexNet'
    date_time = get_datetime()
    # Train
    for epoch in range(1, args.epochs + 1):
        lr, optimizer = exp_lr_scheduler(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, lr))
        print()
        train_loss, train_acc, train_top5 = train_model(train_loader, model, criterion, optimizer, epoch)


    """Test"""

    if 1:
        model = nn.Sequential(*list(model.module.children())[:-1])
	# print(model)
        # model = torch.nn.DataParallel(model)
    	torch.save(model, 'resnet50_trained.pth')
    if 0:
        model = nn.Sequential(*list(model.module.children())[:-1])
	# print(model)
        # model = torch.nn.DataParallel(model)
    	torch.save(model, 'resnet_trained.pth')
    if 0:
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        # model.features = torch.nn.DataParallel(model.features)
    	torch.save(model, 'alexnet_trained.pth')


    score_array = cmc(model)
    print(score_array)

    print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}'.format(
        score_array[0], score_array[4]))

def use_trained_model():

    if 1:
    	model = torch.load('resnet50_trained.pth')
        model = torch.nn.DataParallel(model)
    if 0:
    	model = torch.load('resnet_trained.pth')
    if 0:
    	model = torch.load('alexnet_trained.pth')
    
    if args.cuda:
	model.cuda() 
    # print(model)

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
    # main()
    use_trained_model()
