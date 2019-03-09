# coding: utf-8

# this code is modified from the feature input example code: https://github.com/Hongje/CoVieW2018_temporal_attention-pytorch
#
# Hongje Seong

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import numpy as np
from math import exp
import os
import sys

from pre_extracted_data_loader import pre_extracted_dataset
from models.contextgate import *
from utils import progress_bar
import pdb
import time


train_data_path = '/HDD/place365/data/densenet161/feature/'
test_data_path = '/HDD/place365/data/densenet161/feature/'

model_save_path = '/HDD/place365/weights/densenet161/bilinear/'

batchsize = 256
num_epoch = 1000    # It should less than 10000000
base_learning_rate = 0.01
learning_rate_decay_epoch = 30
learning_rate_decay_rate = 1./10  # It should float & less than 1
model_save_period_epoch = 10

load_weights = False
load_weights_path = '/HDD/place365/weights/ckpt.pt'

input_data_length = [365,1000]
class_num = 365
weight_l2_regularization = 5e-4
data_loader_worker_num = 0 # 2 is default

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc_top1 = 0  # best test accuracy
best_acc_top5 = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)



"""Load Network"""
net = FC1_imagenet_sum(input_data_length=input_data_length, class_num=class_num)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

if load_weights:
    # Load checkpoint.
    print('==> Resuming from trained model..')
    assert os.path.isfile(load_weights_path), 'Error: no weight file! %s'%(load_weights_path)
    checkpoint = torch.load(load_weights_path)
    net.load_state_dict(checkpoint['net'])
    best_acc_top1 = checkpoint['acc_top1']
    best_acc_top5 = checkpoint['acc_top5']
    start_epoch = checkpoint['epoch']
    best_epoch = start_epoch
    best_epoch_top1 = start_epoch
    best_epoch_top5 = start_epoch

training_setting_file = open(os.path.join(model_save_path,'training_settings.txt'), 'a')
training_setting_file.write('----- training options ------\n')
training_setting_file.write('batchsize = %d\n'%batchsize)
training_setting_file.write('num_epoch = %d\n'%num_epoch)
training_setting_file.write('base_learning_rate = %f\n'%base_learning_rate)
training_setting_file.write('learning_rate_decay_epoch = %d\n'%learning_rate_decay_epoch)
training_setting_file.write('learning_rate_decay_rate = %f\n'%learning_rate_decay_rate)
training_setting_file.write('model_save_period_epoch = %d\n'%model_save_period_epoch)
training_setting_file.write('load_weights = %r\n'%load_weights)
training_setting_file.write('start_epoch = %d\n'%start_epoch)
training_setting_file.write('-----------------------------\n')
training_setting_file.write('\n\n')
training_setting_file.close()

criterion = nn.CrossEntropyLoss()



"""Load Dataset"""
transform_train = None
transform_test = None

trainset = pre_extracted_dataset(root=train_data_path, train=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=data_loader_worker_num)

testset = pre_extracted_dataset(root=test_data_path, train=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=data_loader_worker_num)


optimizer = torch.optim.Adam(net.parameters(), lr=base_learning_rate, weight_decay=weight_l2_regularization)
optimizer = torch.optim.SGD(net.parameters(), base_learning_rate, momentum=0.9, weight_decay=weight_l2_regularization)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_epoch, gamma=learning_rate_decay_rate)


# Training
def train(epoch, learning_rate):
    print('\nEpoch: %d' % epoch)
    net.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (data_places, data_imagenet, targets) in enumerate(trainloader):

        data_places, data_imagenet, targets = data_places.to(device), data_imagenet.to(device), targets.to(device)
        outputs = net(data_places, data_imagenet)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy_topk(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), data_places.size(0))
        top1.update(prec1.item(), data_places.size(0))
        top5.update(prec5.item(), data_places.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader), 
                    'Loss: {loss.val:.4f} ({loss.avg:.4f}) | '
                    'Prec@1: {top1.val:.3f} ({top1.avg:.3f}) | '
                    'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    loss=losses, top1=top1, top5=top5))



# Test
def test(epoch):
    global best_acc_top1
    global best_acc_top5
    global best_epoch
    global best_epoch_top1
    global best_epoch_top5
    net.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data_places, data_imagenet, targets) in enumerate(testloader):
            data_places, data_imagenet, targets = data_places.to(device), data_imagenet.to(device), targets.to(device)

            outputs = net(data_places, data_imagenet)
            loss = criterion(outputs, targets)


            prec1, prec5 = accuracy_topk(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), data_places.size(0))
            top1.update(prec1.item(), data_places.size(0))
            top5.update(prec5.item(), data_places.size(0))

            # test_loss += float(loss.item())
            # _, predicted = outputs.max(1)

            # total += targets.size(0)
            # correct += int(predicted.eq(targets).sum().item())
            
            progress_bar(batch_idx, len(testloader), 
                        'Loss: {loss.val:.4f} ({loss.avg:.4f}) | '
                        'Prec@1: {top1.val:.3f} ({top1.avg:.3f}) | '
                        'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                        loss=losses, top1=top1, top5=top5))


    # Save checkpoint.
    acc_top1 = top1.avg
    acc_top5 = top5.avg

    if ((epoch+1) % model_save_period_epoch) == 0:
        print('Saving... periodically')
        state = {
            'net': net.state_dict(),
            'acc_top1': acc_top1,
            'acc_top5': acc_top5,
            'epoch': epoch,
        }
        # torch.save(state, os.path.join(model_save_path, 'weights_%07d.pt'%(epoch+1)))
        torch.save(state, os.path.join(model_save_path, 'weights_latest.pt'))
    
    if acc_top1 > best_acc_top1:
        print('Saving... best test accuracy-top1')
        state = {
            'net': net.state_dict(),
            'acc_top1': acc_top1,
            'acc_top5': acc_top5,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'ckpt.pt'))
        best_acc_top1 = acc_top1
        best_epoch_top1 = epoch
        best_epoch = epoch
    
    if acc_top5 > best_acc_top5:
        print('Saving... best test accuracy-top5')
        state = {
            'net': net.state_dict(),
            'acc_top1': acc_top1,
            'acc_top5': acc_top5,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'ckpt_top5.pt'))
        best_acc_top5 = acc_top5
        best_epoch_top5 = epoch
        best_epoch = epoch

    print('The best test accuracy-top1: %f  epoch: %d'%(best_acc_top1, best_epoch_top1))
    print('The best test accuracy-top5: %f  epoch: %d'%(best_acc_top5, best_epoch_top5))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


for epoch in range(start_epoch, start_epoch+num_epoch):
    scheduler.step()
    train(epoch, base_learning_rate)
    test(epoch)
