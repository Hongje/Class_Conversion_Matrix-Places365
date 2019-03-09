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
parser.add_argument('--arch-places', default='resnet18')
parser.add_argument('--use-custom-model', action='store_false',
                    help='use our custom model (default: true)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=210, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--sub-division', default=1, type=int,
                    metavar='N', help='sub-division size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-places', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-transferlearning', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-contextgating', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
parser.add_argument('--save-path',default='',help='path to save the model')

best_prec1 = 0
best_prec5 = 0
best_epoch1 = 0
best_epoch5 = 0



def main():
    global args
    global best_prec1, best_epoch1
    global best_prec5, best_epoch5
    args = parser.parse_args()
    print args


    assert args.batch_size >= 2, 'Error: batch size must be greater than or equal to 2'
    assert args.sub_division >= 1, 'Error: sub-division size must be greater than or equal to 2'
    assert (args.batch_size % args.sub_division)==0, 'Error: batch size must divided by sub-division!'
    assert (args.batch_size / args.sub_division) >= 2, 'Error: The quotient of the batch size divided by sub-division must be greater than or equal to 2,!'


    # create model
    print("=> creating model '{}'".format(args.arch))
    model_imagenet = models.__dict__[args.arch](pretrained=True)
    
    if args.use_custom_model:
        model_places = models.__dict__[args.arch_places](num_classes=args.num_classes)
    else:
        model_places = custom_models.__dict__[args.arch_places](num_classes=args.num_classes, pretrained=None)



    if args.arch == 'alexnet':
        input_data_length = 4096
    elif args.arch == 'resnet18':
        input_data_length = 512
    elif args.arch == 'resnet50':
        input_data_length = 2048
    elif args.arch == 'densenet161':
        input_data_length = 2208

    model_transferlearning = custom_models.transfer_learning.FC(input_data_length=input_data_length)
    model_contextgating = custom_models.contextgate.FC1_BN_imagenet_sum()



    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    assert os.path.isdir(args.save_path), 'Error: no save directory! %s'%(args.save_path)




    if args.resume is None:
        if args.resume_places:
            if os.path.isfile(args.resume_places):
                print("=> loading checkpoint '{}'".format(args.resume_places))
                checkpoint = torch.load(args.resume_places)

                # #############################################################################
                # model_dict = model.state_dict()
                # pretrained_dict = checkpoint['state_dict']
                # for i in range(len(pretrained_dict.keys())-2):
                #     dict_name = pretrained_dict.keys()[i]
                #     model_dict[dict_name] = pretrained_dict[dict_name]
                # model.load_state_dict(model_dict)
                # #############################################################################


                #############################################################################
                model_dict = model_places.state_dict()
                pretrained_dict = checkpoint['state_dict']
                for i in range(len(pretrained_dict.keys())):
                    dict_name = pretrained_dict.keys()[i]
                    model_dict[dict_name[7:]] = pretrained_dict[dict_name]
                model_places.load_state_dict(model_dict)
                #############################################################################


                # #############################################################################
                # model_dict = model_places.state_dict()
                # pretrained_dict = checkpoint['state_dict']
                # for i in range(len(pretrained_dict.keys())-2):
                #     dict_name = pretrained_dict.keys()[i]
                #     model_dict[dict_name[7:]] = pretrained_dict[dict_name]
                # model_dict['fc.weight'] = pretrained_dict['module.last_conv.weight'].squeeze()
                # model_dict['fc.bias'] = pretrained_dict['module.last_conv.bias']
                # model_places.load_state_dict(model_dict)
                # #############################################################################

                # model.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume_places, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume_places))

        if args.resume_transferlearning:
            #####################################################################################
            # Extract Feature
            if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
                model_places.classifier = model_places.classifier[:-1]
            else:
                new_classifier = nn.Sequential(*list(model_places.fc.children())[:-1])
                model_places.fc = new_classifier
            #####################################################################################

            # Load checkpoint.
            print('==> Resuming from trained model - Transfer Learning Gating..')
            assert os.path.isfile(args.resume_transferlearning), 'Error: no weight file! %s'%(args.resume_transferlearning)
            checkpoint = torch.load(args.resume_transferlearning)

            #############################################################################
            model_dict = model_transferlearning.state_dict()
            pretrained_dict = checkpoint['net']
            for i in range(len(pretrained_dict.keys())):
                dict_name = pretrained_dict.keys()[i]
                model_dict[dict_name[7:]] = pretrained_dict[dict_name]
            model_transferlearning.load_state_dict(model_dict)
            #############################################################################

            # model_contextgating.load_state_dict(checkpoint['net'])

        if args.resume_contextgating:
            # Load checkpoint.
            print('==> Resuming from trained model - Conext Gating..')
            assert os.path.isfile(args.resume_contextgating), 'Error: no weight file! %s'%(args.resume_contextgating)
            checkpoint = torch.load(args.resume_contextgating)

            #############################################################################
            model_dict = model_contextgating.state_dict()
            pretrained_dict = checkpoint['net']
            for i in range(len(pretrained_dict.keys())):
                dict_name = pretrained_dict.keys()[i]
                model_dict[dict_name[7:]] = pretrained_dict[dict_name]
            model_contextgating.load_state_dict(model_dict)
            #############################################################################

            # model_contextgating.load_state_dict(checkpoint['net'])



    # model = custom_models.score_sum(model_imagenet, model_places, model_contextgating)
    model = custom_models.score_sum_transfer_learning(model_imagenet, model_places, model_transferlearning, model_contextgating)

    model = torch.nn.DataParallel(model).cuda()
    print model


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                              weight_decay=args.weight_decay)


    # ##################################################################################
    # # use pre-trained model only cnn parts
    # print('==> load pre-trained conv model weights..')
    # checkpoint = torch.load('/media/ci-mot/a64d4594-a744-46f5-bf47-2e2b72d4ad0d/place365/weights/pre-trained/resnet18_places365.pth.tar')

    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    # model_dict.update(pretrained_dict) 
    # model.load_state_dict(model_dict)

    # state = {
    #     'epoch': 0,
    #     'arch': args.arch,
    #     'state_dict': model.state_dict(),
    #     'best_prec1': 0,
    #     'best_prec5': 0,
    #     'best_epoch1': 0,
    #     'best_epoch5': 0,
    # }
    # torch.save(state, args.resume)
    # pdb.set_trace()
    # ##################################################################################

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

            # best_prec1 = 0
            # best_prec5 = 0
            # best_epoch1 = 0
            # best_epoch5 = 0
            
            # #############################################################################
            # import re
            # dot_error = checkpoint['state_dict']
            # names_dot_error = dot_error.keys()
            # model_dict = model.state_dict()
            # correct_list = model_dict.keys()
            # correct_list_nodot = []
            # for i in range(len(correct_list)):
            #     correct_list_nodot.append(re.sub(r'[^\w\s]','',correct_list[i]))

            # for name in names_dot_error:
            #     name_nodot = re.sub(r'[^\w\s]','',name)
            #     for i in range(len(correct_list)):
            #         if name_nodot == correct_list_nodot[i]:
            #             dot_error[correct_list[i]] = dot_error.pop(name)
            #             break

            # model.load_state_dict(dot_error)
            # #############################################################################


            # #############################################################################
            # model_dict = model.state_dict()
            # pretrained_dict = checkpoint['state_dict']
            # for i in range(len(pretrained_dict.keys())-2):
            #     dict_name = pretrained_dict.keys()[i]
            #     model_dict[dict_name] = pretrained_dict[dict_name]
            # model.load_state_dict(model_dict)
            # #############################################################################

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size // args.sub_division, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size // (args.sub_division*2), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    
    # evaluate on validation set at first
    prec1, prec5 = validate(val_loader, model, criterion)

    print('The best test accuracy-top1: %f  epoch: %d'%(best_prec1, best_epoch1))
    print('The best test accuracy-top5: %f  epoch: %d'%(best_prec5, best_epoch5))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        if prec1 > best_prec1:
            best_prec1 = prec1
            best_epoch1 = epoch
        if prec5 > best_prec5:
            best_prec5 = prec5
            best_epoch5 = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'best_epoch1': best_epoch1,
            'best_epoch5': best_epoch5,
            'optimizer_dict': optimizer.state_dict(),
        }, is_best, args.arch.lower())

        print('The best test accuracy-top1: %f  epoch: %d'%(best_prec1, best_epoch1))
        print('The best test accuracy-top5: %f  epoch: %d'%(best_prec5, best_epoch5))


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


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

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

    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


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
