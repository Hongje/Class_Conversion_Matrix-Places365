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
from models.CCM import *
from utils import progress_bar
import pdb
import time


arch = 'alexnet'
arch = arch + '_transfer_learning'
train_data_path = os.path.join('/HDD/place365/data/', arch, 'score/')
test_data_path = os.path.join('/HDD/place365/data/', arch, 'score/')

model_save_path = 'total_Adam_lr-1_batch2048'
model_save_path = os.path.join('/HDD/place365/weights/', arch, model_save_path)

batchsize = 2048
num_epoch = 1000    # It should less than 10000000
base_learning_rate = 0.1
learning_rate_decay_epoch = 30
learning_rate_decay_rate = 1./10  # It should float & less than 1
model_save_period_epoch = 10


# train_models_num = 6


load_weights = False
load_weights_path = '/HDD/weights/ckpt.pt'

input_data_length = [365,1000]
class_num = 365
weight_l2_regularization = 5e-4
data_loader_worker_num = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc_top1 = 0  # best test accuracy
best_acc_top5 = 0
best_acc_top1_avg = 0
best_acc_top5_avg = 0
best_top1_index = 0
best_top5_index = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)



"""Load Network"""
net = []
# for i in range(train_models_num):
#     net.append(FC1(input_data_length=input_data_length, class_num=class_num))

net.append(FC1_only_imagenet(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_only_imagenet(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_only_place(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_only_place(input_data_length=input_data_length, class_num=class_num))

net.append(FC1_imagenet_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenet_sum(input_data_length=input_data_length, class_num=class_num))

net.append(FC1_imagenetReLU_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenetReLU_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_imagenetReLU_bias_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenetReLU_bias_sum(input_data_length=input_data_length, class_num=class_num))

net.append(FC1_imagenet_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenet_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_imagenet_Sigmoidweighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenet_Sigmoidweighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_imagenetReLU_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenetReLU_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_imagenetReLU_Sigmoidweighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenetReLU_Sigmoidweighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_imagenetReLU_bias_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenetReLU_bias_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_imagenetReLU_bias_Sigmoidweighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC1_BN_imagenetReLU_bias_Sigmoidweighted_sum(input_data_length=input_data_length, class_num=class_num))

net.append(FC_imagenet_FC_place_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC_BN_imagenet_FC_BN_place_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC_imagenet_FC_place_weighted_sum(input_data_length=input_data_length, class_num=class_num))
net.append(FC_BN_imagenet_FC_BN_place_weighted_sum(input_data_length=input_data_length, class_num=class_num))

train_models_num = len(net)


best_acc_top1_models = []
best_acc_top5_models = []
best_epoch_models = []
best_epoch_top1_models = []
best_epoch_top5_models = []
for i in range(train_models_num):
    best_acc_top1_models.append(0)
    best_acc_top5_models.append(0)
    best_epoch_models.append(0)
    best_epoch_top1_models.append(0)
    best_epoch_top5_models.append(0)


for i in range(train_models_num):
    net[i] = net[i].to(device)
if device == 'cuda':
    for i in range(train_models_num):
        net[i] = torch.nn.DataParallel(net[i])
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

if load_weights:
    # Load checkpoint.
    print('==> Resuming from trained model..')
    assert os.path.isfile(load_weights_path), 'Error: no weight file! %s'%(load_weights_path)
    checkpoint = torch.load(load_weights_path)
    for i in range(train_models_num):
        net[i].load_state_dict(checkpoint['net'])
        best_epoch_top1[i] = start_epoch
        best_epoch_top5[i] = start_epoch
    best_acc_top1 = checkpoint['acc_top1']
    best_acc_top5 = checkpoint['acc_top5']
    best_epoch = start_epoch
    start_epoch = checkpoint['epoch']

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


criterion = []
for i in range(train_models_num):
    criterion.append(nn.CrossEntropyLoss())



"""Load Dataset"""
transform_train = None
transform_test = None

trainset = pre_extracted_dataset(root=train_data_path, train=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=data_loader_worker_num)

testset = pre_extracted_dataset(root=test_data_path, train=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=data_loader_worker_num)

optimizer = []
scheduler = []
for i in range(train_models_num):
    optimizer.append(torch.optim.Adam(net[i].parameters(), lr=base_learning_rate, weight_decay=weight_l2_regularization))
    # optimizer.append(torch.optim.SGD(net[i].parameters(), base_learning_rate, momentum=0.9, weight_decay=weight_l2_regularization))
for i in range(train_models_num):
    scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[i], step_size=learning_rate_decay_epoch, gamma=learning_rate_decay_rate))


# Training
def train(epoch, learning_rate):
    print('\nEpoch: %d' % epoch)

    losses = []
    top1 = []
    top5 = []
    for i in range(train_models_num):
        net[i].train()
        losses.append(AverageMeter())
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    losses_global = AverageMeter()
    top1_global = AverageMeter()
    top5_global = AverageMeter()

    for batch_idx, (data_places, data_imagenet, targets) in enumerate(trainloader):

        data_places, data_imagenet, targets = data_places.to(device), data_imagenet.to(device), targets.to(device)


        outputs = []
        for i in range(train_models_num):
            outputs.append(net[i](data_places, data_imagenet))
            
        loss = []
        for i in range(train_models_num):
            loss.append(criterion[i](outputs[i], targets))

        for i in range(train_models_num):
            prec1, prec5 = accuracy_topk(outputs[i].data, targets, topk=(1, 5))
            losses[i].update(loss[i].item(), data_places.size(0))
            top1[i].update(prec1.item(), data_places.size(0))
            top5[i].update(prec5.item(), data_places.size(0))

            losses_global.update(loss[i].item(), data_places.size(0))
            top1_global.update(prec1.item(), data_places.size(0))
            top5_global.update(prec5.item(), data_places.size(0))

        # compute gradient and do Adam step
        for i in range(train_models_num):
            optimizer[i].zero_grad()
            loss[i].backward()
            optimizer[i].step()

        progress_bar(batch_idx, len(trainloader), 
                    'Loss: {loss.avg:.4f} | '
                    'Prec@1: {top1.avg:.3f} | '
                    'Prec@5: {top5.avg:.3f}'.format(
                    loss=losses_global, top1=top1_global, top5=top5_global))



# Test
def test(epoch):
    global best_acc_top1
    global best_acc_top5
    global best_acc_top1_avg
    global best_acc_top5_avg
    global best_top1_index
    global best_top5_index
    global best_epoch
    global best_epoch_top1
    global best_epoch_top5

    losses = []
    top1 = []
    top5 = []
    for i in range(train_models_num):
        net[i].eval()
        losses.append(AverageMeter())
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    
    losses_global = AverageMeter()
    top1_global = AverageMeter()
    top5_global = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data_places, data_imagenet, targets) in enumerate(testloader):
            data_places, data_imagenet, targets = data_places.to(device), data_imagenet.to(device), targets.to(device)

            outputs = []
            for i in range(train_models_num):
                outputs.append(net[i](data_places, data_imagenet))

            loss = []
            for i in range(train_models_num):
                loss.append(criterion[i](outputs[i], targets))

            for i in range(train_models_num):
                prec1, prec5 = accuracy_topk(outputs[i].data, targets, topk=(1, 5))
                losses[i].update(loss[i].item(), data_places.size(0))
                top1[i].update(prec1.item(), data_places.size(0))
                top5[i].update(prec5.item(), data_places.size(0))

                losses_global.update(loss[i].item(), data_places.size(0))
                top1_global.update(prec1.item(), data_places.size(0))
                top5_global.update(prec5.item(), data_places.size(0))

            
            progress_bar(batch_idx, len(testloader), 
                        'Loss: {loss.avg:.4f} | '
                        'Prec@1: {top1.avg:.3f} | '
                        'Prec@5: {top5.avg:.3f}'.format(
                        loss=losses_global, top1=top1_global, top5=top5_global))


    # Save checkpoint.
    acc_top1 = []
    acc_top5 = []
    current_best = []
    for i in range(train_models_num):
        acc_top1.append(top1[i].avg)
        acc_top5.append(top5[i].avg)
        current_best.append(False)


    for i in range(train_models_num):
        if best_acc_top1_models[i] < acc_top1[i]:
            best_acc_top1_models[i] = acc_top1[i]
            best_epoch_top1_models[i] = epoch
            current_best[i] = True
        if best_acc_top5_models[i] < acc_top5[i]:
            best_acc_top5_models[i] = acc_top5[i]
            best_epoch_top5_models[i] = epoch
    best_acc_top1_avg = sum(best_acc_top1_models) / float(len(best_acc_top1_models))
    best_acc_top5_avg = sum(best_acc_top5_models) / float(len(best_acc_top5_models))

    max_acc_top1_index = acc_top1.index(max(acc_top1))
    max_acc_top5_index = acc_top5.index(max(acc_top5))
    max_acc_top1_value = max(acc_top1)
    max_acc_top5_value = max(acc_top5)


    if ((epoch+1) % model_save_period_epoch) == 0:
        for i in range(train_models_num):
            state = {
                'net': net[i].state_dict(),
                'acc_top1': acc_top1[i],
                'acc_top5': acc_top5[i],
                'epoch': epoch,
            }
            torch.save(state, os.path.join(model_save_path, 'weights_idx_%04d_lastest.pt'%(i)))


    if max_acc_top1_value > best_acc_top1:
        print('Saving... best test accuracy-top1')
        state = {
            'net': net[max_acc_top1_index].state_dict(),
            'acc_top1': max_acc_top1_value,
            'acc_top5': acc_top5[max_acc_top1_index],
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'ckpt.pt'))
        # torch.save(state, os.path.join(model_save_path, 'weights_%07d.pt'%(epoch+1)))
        best_acc_top1 = max_acc_top1_value
        best_top1_index = max_acc_top1_index
        best_epoch_top1 = epoch
        best_epoch = epoch

    if max_acc_top5_value > best_acc_top5:
        print('Saving... best test accuracy-top5')
        state = {
            'net': net[max_acc_top1_index].state_dict(),
            'acc_top1': acc_top1[i],
            'acc_top5': acc_top5[i],
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'ckpt_top5.pt'))
        # torch.save(state, os.path.join(model_save_path, 'weights_%07d.pt'%(epoch+1)))
        best_acc_top5 = max_acc_top5_value
        best_top5_index = max_acc_top5_index
        best_epoch_top5 = epoch
        best_epoch = epoch
    
    for i in range(train_models_num):
        if current_best[i]:
            print('Saving... best test accuracy-top1 model: %d'%(i))
            state = {
                'net': net[i].state_dict(),
                'acc_top1': max_acc_top1_value,
                'acc_top5': acc_top5[i],
                'epoch': epoch,
            }
            torch.save(state, os.path.join(model_save_path, 'ckpt_top1_idx_%04d.pt'%(i)))


    print('The best test accuracy-top1: %f  epoch: %d  index: %d'%(best_acc_top1, best_epoch_top1, best_top1_index))
    print('The best test accuracy-top5: %f  epoch: %d  index: %d'%(best_acc_top5, best_epoch_top5, best_top5_index))
    print('The best test avg acc-top1: %f%%  acc-top5: %f%%'%(best_acc_top1_avg, best_acc_top5_avg))
    for i in range(train_models_num):
        print('%03d_top1_%.3f%%_epoch_%d_top5_%.4f%%_epoch_%d'%(
              i, best_acc_top1_models[i], best_epoch_top1_models[i], best_acc_top5_models[i], best_epoch_top5_models[i]))
    print('')
    for i in range(train_models_num):
        print('model number: %03d | loss: %.3f | best acc: (%.3f%%, %.3f%%)  current acc: (%.3f%%, %.3f%%) | best epoch: (%d, %d)'%(
              i, losses[i].avg, best_acc_top1_models[i], best_acc_top5_models[i], acc_top1[i], acc_top5[i], best_epoch_top1_models[i], best_epoch_top5_models[i]))



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
    for i in range(train_models_num):
        scheduler[i].step()
    train(epoch, base_learning_rate)
    test(epoch)
