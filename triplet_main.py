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
import scipy
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
parser.add_argument('--train-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
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
def range_except_k(n, end, start = 0):
    return range(start, n) + range(n+1, end)

def _get_triplet_data():
    with h5py.File('triplet-cuhk-03.h5', 'r') as ff:
        class_num = len(ff['a']['train'].keys())
        temp_a = []
        temp_b = []
        temp_id = []
        for i in range(class_num):
	    len1 = len(ff['a']['train'][str(i)])
	    len2 = len(ff['b']['train'][str(i)])
            if len1 < 5 or len2 < 5:
                class_num -= 1
            if len1 >= 5 and len2 >= 5:
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

        pair_num = 4000
        triplet_temp = torch.FloatTensor(pair_num, 3, camera1_dataset.size(2), camera1_dataset.size(3), camera1_dataset.size(3)).zero_()
        triplet_id_temp = torch.LongTensor(pair_num, 3)

        for i in range(pair_num):
            k = random.randint(0, class_num-1)
            range_no_k = range_except_k(k, class_num)
            k1 = random.choice(range_no_k)
            j0 = random.randint(0, 4)
            j1 = random.randint(0, 4)
            j2 = random.randint(0, 4)
            triplet_temp[i][0] = camera1_dataset[k][j0]
            triplet_id_temp[i][0] = k
            triplet_temp[i][1] = camera2_dataset[k][j1]
            triplet_id_temp[i][1] = k
            triplet_temp[i][2] = camera1_dataset[k1][j2]
            triplet_id_temp[i][2] = k1

        triplet_dataset = triplet_temp
        triplet_label = triplet_id_temp
        print('class_num: ', class_num)
        return triplet_dataset, triplet_label


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

        return (camere1, camere2)


def train_model(train_loader, model, criterion, optimizer, epoch):

    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        triplet_pair = torch.split(inputs, 1, 1)
        anchor = triplet_pair[0].resize_(args.train_batch_size, inputs.size(2), inputs.size(3), inputs.size(4))
        positive = triplet_pair[1].resize_(args.train_batch_size, inputs.size(2), inputs.size(3), inputs.size(4))
        negative = triplet_pair[2].resize_(args.train_batch_size, inputs.size(2), inputs.size(3), inputs.size(4))
        if batch_idx == 2:
            image1, image2, image3 = anchor.numpy(), positive.numpy(), negative.numpy()
            img1, img2, img3 = image1.transpose(0,2,3,1), image2.transpose(0,2,3,1), image3.transpose(0,2,3,1)
            file_path = 'tripletimage/'
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            scipy.misc.imsave(directory+'/'+str(0)+'.png', img1[10])
            scipy.misc.imsave(directory+'/'+str(1)+'.png', img2[10])
            scipy.misc.imsave(directory+'/'+str(2)+'.png', img3[10])
            sys.exit('exit')
        triplet_label_pair = torch.split(targets, 1, 1)
        anchor_label = triplet_label_pair[0].resize_(args.train_batch_size)
        positive_label = triplet_label_pair[1].resize_(args.train_batch_size)
        negative_label = triplet_label_pair[2].resize_(args.train_batch_size)

        anchor = anchor.float()  # with size of (batch_size * 3 * 224 * 224)
        positive = positive.float()
        negative = negative.float()
        anchor_label.long()
        positive_label.long()
        negative_label.long()
        if args.cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)
        if args.cuda:
            anchor_label, positive_label, negative_label = anchor_label.cuda(), positive_label.cuda(), negative_label.cuda()
        anchor_label, positive_label, negative_label = Variable(anchor_label), Variable(positive_label), Variable(negative_label)

        if 0:
            print(anchor[0])
            print(positive[0])
            print(negative[0])
        # compute output
        outputs1, outputs2, outputs3 = model(anchor), model(positive), model(negative)
        # print('anchor size: ', anchor.size())
        # print('output11 size: ', outputs1.size())

        if 0:
            print(outputs1)
            print(outputs2)
            print(outputs3)
            sys.exit('exit')

        loss = criterion(outputs1, outputs2, outputs3)
        # print(loss.data)
        # sys.exit('exit')

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

    model.eval()
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



model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
model.features = torch.nn.DataParallel(model.features)
if args.cuda:
    model.cuda()

ori_model = models.alexnet(pretrained=True)
class AlexNetNoClassifier(nn.Module):
        def __init__(self):
            super(AlexNetNoClassifier, self).__init__()
            self.features = nn.Sequential(
                *list(ori_model.features.children())[:]
                # *list(original_model.features.children())
            )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            return x

new_model = AlexNetNoClassifier()
new_model.features = torch.nn.DataParallel(new_model.features)
if args.cuda:
    new_model.cuda()


def main():

    original_model = models.alexnet(pretrained=True)
    # print(original_model)
    original_model.classifier = nn.Sequential(*list(original_model.classifier.children())[:-6])
    mymodel = original_model
    mymodel.classifier._modules['0'] = nn.Dropout(p=0.0001)
    mymodel.features = torch.nn.DataParallel(mymodel.features)
    if args.cuda:
        mymodel.cuda()


    triplet_dataset, triplet_label = _get_triplet_data()
    print('train data  size: ', triplet_dataset.size())
    print('train target size', triplet_label.size())
    train_data = data_utils.TensorDataset(triplet_dataset, triplet_label)
    train_loader = data_utils.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

    cudnn.benchmark = True

    optimizer = optim.SGD(new_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    if args.cuda:
        criterion = nn.TripletMarginLoss(margin=1.0, p=2).cuda()


    title = 'CUHK03-AlexNet'

    # Train
    for epoch in range(1, args.epochs + 1):
        lr, optimizer = exp_lr_scheduler(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, lr))
        print()
        # train_model(train_loader, new_model, criterion, optimizer, epoch)
        train_model(train_loader, model, criterion, optimizer, epoch)

    # Test

    torch.save(mymodel, 'mymodel_triplet_trained.pth')

    score_array = cmc(mymodel)
    print(score_array)

    print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}'.format(
        score_array[0], score_array[4]))

def use_trained_model():

    model = torch.load('mymodel_triplet_trained.pth')

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
