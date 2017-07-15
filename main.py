from __future__ import print_function
import argparse
import os
import h5py
import sys
import time
import datetime
import argparse
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision import models
import torch.utils.data as data_utils
from cuhk03_alexnet import AlexNet
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
parser.add_argument('--train-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
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
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Data loading
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
normalize = transforms.Normalize(mean = [ 0.367, 0.362, 0.357 ],
                                 std = [ 0.244, 0.247, 0.249 ])
traindir = os.path.join('.', 'train')
valdir = os.path.join('.', 'val')
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize,
                   ])),
    batch_size=args.train_batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
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
    return (losses.avg, top1.avg, top5.avg)

def test(model, criterion, epoch):
    global best_acc
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   batch_idx, len(val_loader), loss=losses,
                   top1=top1, top5=top5))
            print()
    print('Test batch size: %s' %(args.test_batch_size))
    print('Top1(test) : {:.3f}%\tTop5(test) : {:.3f}%'.format(
        top1.avg, top5.avg))
    print('Test Average Loss: {:.4f}'.format(losses.avg))
    print()
    return (losses.avg, top1.avg, top5.avg)


best_acc = 0  # best test accuracy

def main():
    global best_acc
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    model_name = ''
    pretrain = ''
    classes_num = [843, 1467, 624]
    if 0:
        model = models.vgg11(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, classes_num[0])
        # model.classifier._modules['6'].weight.data.normal_(0.0, 0.3)
        # model.classifier._modules['6'].bias.data.zero_()
        model.features = torch.nn.DataParallel(model.features)
        pretrain = '1'
        model_name = 'vgg11'

    if 0:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes_num[0])
        # model.fc.weight.data.normal_(0.0, 0.3)
        # model.fc.bias.data.fill_(0)
        model = torch.nn.DataParallel(model)
        pretrain = '1'
        model_name = 'resnet18'

    if 1:
        model = models.alexnet(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, classes_num[0])
        # model.classifier._modules['6'].weight.data.normal_(0.0, 0.3)
        import torch.nn.init as init
        # init.constant(model.classifier._modules['6'].bias, 0.0)
        model.features = torch.nn.DataParallel(model.features)
        pretrain = '1'
        model_name = 'alexnet'

    if args.cuda:
        model.cuda()
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()

    title = 'CUHK03-AlexNet'
    now_datetime = str(datetime.datetime.now())
    array0 = now_datetime.split(' ')
    yymmdd = array0[0]
    time_array = array0[1].split(':')
    hour_min = time_array[0]+time_array[1]
    log_filename = 'log-'+model_name+'-'+pretrain+'-'+yymmdd+'-'+hour_min+'.txt'
    logger = Logger(os.path.join(args.checkpoint, log_filename), title=title)
    # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train Top5', 'Valid Top5'])
    # Train and val
    for epoch in range(1, args.epochs + 1):
        lr, optimizer = exp_lr_scheduler(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, lr))
        print()
        train_loss, train_acc, train_top5 = train(model, criterion, optimizer, epoch)
        test_loss, test_acc, test_top5 = test(model, criterion, epoch)

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc, train_top5, test_top5])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        # save_checkpoint({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'acc': test_acc,
        #         'best_acc': best_acc,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, checkpoint=args.checkpoint)

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best test acc: {:.3f}'.format(best_acc))


def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=20):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    # if epoch % lr_decay_epoch == 0:
        # print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, optimizer

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()


# model = models.vgg11(pretrained=True)
# model.fc = nn.Linear(4096, 843)
# m = model.classifier._modules['6']
# m = nn.Linear(4096, 843)
# m.weight.data.normal_(0.0, 0.3)
# m.bias.data.zero_()

# model = models.resnet18(pretrained=True)
# m = list(model.children())[-1]
# num_ftrs = model.fc.in_features
# m = nn.Linear(num_ftrs, 843)
# m.weight.data.normal_(0.0, 0.3)
# m.bias.data.zero_()
# m.bias.data.fill_(0)

# model = AlexNet()
