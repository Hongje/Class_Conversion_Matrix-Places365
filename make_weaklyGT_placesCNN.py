# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

# this code is modified from the Places2 pytorch example code: https://github.com/CSAILVision/places365
#
# Hongje Seong

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import models

import wideresnet
import pdb
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('-data', metavar='DIR', default='/HDD/place365/data/places365standard_easyformat/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/HDD/place365/weights/pre-trained/alexnet_places365.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to run')
parser.add_argument('--save-path',default='/media/ci-mot/a64d4594-a744-46f5-bf47-2e2b72d4ad0d/place365/data/',help='path to save the data')
parser.add_argument('--pretrained-dataset',default='places365',help='which dataset to the pre-trained model is trained')
# parser.add_argument('--pretrained-dataset',default='imagenet',help='which dataset to the pre-trained model is trained')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print args


    path = os.path.join(args.save_path, args.arch, args.pretrained_dataset)
    if not os.path.isdir(path):
        os.mkdir(path)
    assert os.path.isdir(path), 'Error: no save directory! %s'%(path)



    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.pretrained_dataset == 'places365':
        if args.arch.lower().startswith('wideresnet'):
            # a customized resnet model with last feature map size as 14x14 for better class activation mapping
            model  = wideresnet.resnet50(num_classes=args.num_classes)
        else:
            model = models.__dict__[args.arch](num_classes=args.num_classes)

    elif args.pretrained_dataset == 'imagenet':
        model = models.__dict__[args.arch](pretrained=True)
        # model = models.__dict__[args.arch](num_classes=1000, pretrained='imagenet')

    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
        # model = model

    
    print model

    # optionally resume from a checkpoint
    if args.resume and args.pretrained_dataset == 'places365':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    

    cudnn.benchmark = True

    #####################################################################################
    # Extract Feature
    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.classifier = model.classifier[:-1]
    else:
        new_classifier = nn.Sequential(*list(model.module.fc.children())[:-1])
        model.module.fc = new_classifier
    #####################################################################################

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = None


    train(train_loader, model, criterion, optimizer, args.start_epoch)
    prec1 = validate(val_loader, model, criterion)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    N_data = []
    N_target = []

    # # switch to train mode
    # model.train()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            output = model(input_var)
            # output, _ = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # # compute gradient and do SGD step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                # pdb.set_trace()

            N_data.append(output.cpu().numpy())
            N_target.append(target.cpu().numpy())
    
    N_data = np.concatenate(N_data)
    N_target = np.concatenate(N_target)
    data_save_path = os.path.join(args.save_path, args.arch, args.pretrained_dataset, 'train.npz')
    np.savez(data_save_path, data=N_data, label=N_target)



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    N_data = []
    N_target = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            # output, _ = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            
            N_data.append(output.cpu().numpy())
            N_target.append(target.cpu().numpy())

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    N_data = np.concatenate(N_data)
    N_target = np.concatenate(N_target)
    data_save_path = os.path.join(args.save_path, args.arch, args.pretrained_dataset, 'validate.npz')
    np.savez(data_save_path, data=N_data, label=N_target)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename1 = filename + '_latest.pth.tar'
    filename2 = filename + '_best.pth.tar'
    filename1 = os.path.join(args.save_path, filename1)
    filename2 = os.path.join(args.save_path, filename2)
    torch.save(state, filename1)
    if is_best:
        torch.save(state, filename2)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
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


if __name__ == '__main__':
    main()
