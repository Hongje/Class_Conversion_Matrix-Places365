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

from data_loader_places_large import places_dataset
import wideresnet

from utils import progress_bar
from utils import format_time
import models as custom_models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('-data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--use-custom-model', action='store_true',
                    help='use our custom model (default: true)')
parser.add_argument('--dataset-format',default='places365',help='dataset format: places365 or imagenet')
parser.add_argument('--ten-crop-validation', action='store_true',
                    help='useten-crop-validation (default: true)')
parser.add_argument('--class-balanced-sampling', action='store_true',
                    help='use class-balanced-sampling for training (default: true)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=210, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--sub-division', default=1, type=int,
                    metavar='N', help='sub-division size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
parser.add_argument('--save-path',default='',help='path to save the model')

best_prec1 = 0
best_prec5 = 0
best_epoch1 = 0
best_epoch5 = 0

best_prec1_10crop = 0
best_prec5_10crop = 0
best_epoch1_10crop = 0
best_epoch5_10crop = 0



def main():
    global args
    global best_prec1, best_epoch1
    global best_prec5, best_epoch5
    global best_prec1_10crop, best_epoch1_10crop
    global best_prec5_10crop, best_epoch5_10crop
    args = parser.parse_args()
    print args


    assert args.batch_size >= 2, 'Error: batch size must be greater than or equal to 2'
    assert args.sub_division >= 1, 'Error: sub-division size must be greater than or equal to 2'
    assert (args.batch_size % args.sub_division)==0, 'Error: batch size must divided by sub-division!'
    assert (args.batch_size / args.sub_division) >= 2, 'Error: The quotient of the batch size divided by sub-division must be greater than or equal to 2,!'


    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.use_custom_model:
        if args.arch.lower().startswith('wideresnet'):
            # a customized resnet model with last feature map size as 14x14 for better class activation mapping
            model  = wideresnet.resnet50(num_classes=args.num_classes)
        else:
            model = models.__dict__[args.arch](num_classes=args.num_classes)
    else:
        model = custom_models.__dict__[args.arch](num_classes=args.num_classes, pretrained=None)

    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    # print model


    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    assert os.path.isdir(args.save_path), 'Error: no save directory! %s'%(args.save_path)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                              weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            best_epoch1 = checkpoint['best_epoch1']
            best_epoch5 = checkpoint['best_epoch5']
            best_prec1_10crop = checkpoint['best_prec1_10crop']
            best_prec5_10crop = checkpoint['best_prec5_10crop']
            best_epoch1_10crop = checkpoint['best_epoch1_10crop']
            best_epoch5_10crop = checkpoint['best_epoch5_10crop']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_data_transforms_10crop = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])

    if args.dataset_format == 'places365':
        dataloader = places_dataset
        traindir = os.path.join(args.data, 'data_large')
        valdir = os.path.join(args.data, 'val_large')
        filelist = os.path.join(args.data, 'filelist')

        train_data_loader = places_dataset(traindir, filelist, train_data_transforms, train=True)
        val_data_loader = places_dataset(valdir, filelist, val_data_transforms, train=False)

        if not args.ten_crop_validation:
            val_data_loader_10crop = places_dataset(valdir, filelist, val_data_transforms_10crop, train=False)

    elif args.dataset_format == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_data_loader = datasets.ImageFolder(traindir, train_data_transforms)
        val_data_loader = datasets.ImageFolder(valdir, val_data_transforms)

        if not args.ten_crop_validation:
            val_data_loader_10crop = datasets.ImageFolder(valdir, val_data_transforms_10crop)

    if not args.class_balanced_sampling:
        sampler_weights = make_weights_for_balanced_classes(train_data_loader)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weights, len(sampler_weights))
    else:
        train_sampler = None
        

    train_loader = torch.utils.data.DataLoader(
        train_data_loader,
        batch_size=args.batch_size // args.sub_division, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_data_loader,
        batch_size=args.batch_size // (args.sub_division*2), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not args.ten_crop_validation:
        val_loader_10crop = torch.utils.data.DataLoader(
            val_data_loader_10crop,
            batch_size=args.batch_size // (args.sub_division*2*8), shuffle=False,
            num_workers=args.workers, pin_memory=True)
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, tencrop=False)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        if prec1 > best_prec1:
            best_prec1 = prec1
            best_epoch1 = epoch
        if prec5 > best_prec5:
            best_prec5 = prec5
            best_epoch5 = epoch

        if not args.class_balanced_sampling:
            prec1_10crop, prec5_10crop = validate(val_loader_10crop, model, criterion, tencrop=True)

            # remember best prec@1 and save checkpoint
            is_best_10crop = prec1_10crop > best_prec1_10crop
            # best_prec1 = max(prec1, best_prec1)
            if prec1_10crop > best_prec1_10crop:
                best_prec1_10crop = prec1_10crop
                best_epoch1_10crop = epoch
            if prec5_10crop > best_prec5_10crop:
                best_prec5_10crop = prec5_10crop
                best_epoch5_10crop = epoch
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'best_epoch1': best_epoch1,
                'best_epoch5': best_epoch5,
                'best_prec1_10crop': best_prec1_10crop,
                'best_prec5_10crop': best_prec5_10crop,
                'best_epoch1_10crop': best_epoch1_10crop,
                'best_epoch5_10crop': best_epoch5_10crop,
                'optimizer_dict': optimizer.state_dict(),
            }

            save_checkpoint(state, is_best_10crop, args.arch.lower(), tencrop=True)

        if not args.class_balanced_sampling:
            save_checkpoint(state, is_best, args.arch.lower(), tencrop=False)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'best_epoch1': best_epoch1,
                'best_epoch5': best_epoch5,
                'optimizer_dict': optimizer.state_dict(),
            }, is_best, args.arch.lower(), tencrop=False)

        print('The best test accuracy-top1: %f  epoch: %d'%(best_prec1, best_epoch1))
        print('The best test accuracy-top5: %f  epoch: %d'%(best_prec5, best_epoch5))

        print('The best test 10crop accuracy-top1: %f  epoch: %d'%(best_prec1_10crop, best_epoch1_10crop))
        print('The best test 10crop accuracy-top5: %f  epoch: %d'%(best_prec5_10crop, best_epoch5_10crop))


def train(train_loader, model, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        (loss / args.sub_division).backward()
        # optimizer.step()
        if (i+1) % args.sub_division == 0:
            optimizer.step()
            optimizer.zero_grad()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #            epoch, i, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1, top5=top5))
        progress_bar(i, len(train_loader), 
                     'Data: {data_time} | '
                     'Target Loss: {loss.val:.4f} ({loss.avg:.4f}) | '
                     'Prec@1: {top1.val:.3f} ({top1.avg:.3f}) | '
                     'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                     data_time=format_time(data_time.sum),
                     loss=losses, top1=top1, top5=top5))
    if (i+1)%args.sub_division != 0:
        optimizer.step()
        optimizer.zero_grad()


def validate(val_loader, model, criterion, tencrop=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            if tencrop:
                bs, ncrops, c, h, w = input_var.size()
                # compute output
                temp_output = model(input_var.view(-1, c, h, w))
                output = temp_output.view(bs, ncrops, -1).mean(1)
            else:
                # compute output
                output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #            i, len(val_loader), batch_time=batch_time, loss=losses,
            #            top1=top1, top5=top5))
            progress_bar(i, len(val_loader), 
                        'Data: {data_time} | '
                        'Target Loss: {loss.val:.4f} ({loss.avg:.4f}) | '
                        'Prec@1: {top1.val:.3f} ({top1.avg:.3f}) | '
                        'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                        data_time=format_time(data_time.sum),
                        loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', tencrop=False):
    filename1 = filename + '_latest.pth.tar'
    if tencrop:
        filename2 = filename + '_best_10crop.pth.tar'
    else:
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
    lr = args.lr * (0.1 ** (epoch // 20))
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


def make_weights_for_balanced_classes(dataset):
    images = dataset.imgs
    nclasses = len(dataset.classes)
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

if __name__ == '__main__':
    main()
