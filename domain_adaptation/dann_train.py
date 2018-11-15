'''
Training of fly images using Domain Adversarial Neural Networks
based on: Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." 
          The Journal of Machine Learning Research 17.1 (2016): 2096-2030.
'''

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from skimage.draw import circle

import numpy as np
import math
import os.path
import getpass
import pdb

# Global Variables
global best_prec1, best_domain_acc
best_prec1 = 0
best_domain_acc = 0

'''
# launch ipython debugger on exception
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)
'''

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# required for random masking
class HolePunch(object):
    """Randomly blank out a few holes."""

    def __init__(self, n_holes, radius, square=False):
        self.n_holes = n_holes
        self.radius = radius
        self.square = square

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Image with a few holes blanked out.
        """
        h = img.size(1)
        w = img.size(2)

        blobs = np.ones((h, w), np.float32)
        if self.square:
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.radius, 0, h)
                y2 = np.clip(y + self.radius, 0, h)
                x1 = np.clip(x - self.radius, 0, w)
                x2 = np.clip(x + self.radius, 0, w)

                blobs[y1: y2, x1: x2] = 0.
        else:
            for n in range(self.n_holes):
                rr, cc = circle(np.random.randint(h), np.random.randint(w), self.radius, shape=(h, w))
                blobs[rr, cc] = 0.

        blobs = torch.from_numpy(blobs)
        blobs = blobs.expand_as(img)
        img = img * blobs

        return img


'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FEATURE EXTRACTOR (RESNET CLASS DEFINITIONS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, no_classes=8):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(512 * block.expansion, no_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
END OF FEATURE EXTRACTOR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
CLASS CLASSIFIER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

class class_classifier(nn.Module):

    def __init__(self, block, layers, no_classes=8):
        self.inplanes = 256
        super(class_classifier, self).__init__()
        self.layer = self._make_layer(block, 512, layers[0], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, no_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
END OF CLASS CLASSIFIER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DOMAIN CLASSIFIER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

class GradReverse(Function):

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (-1*grad_output)

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(14*14*256, 256) 
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = GradReverse()(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = F.leaky_relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return F.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
END OF DOMAIN CLASSIFIER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--descriptor', default='test', type=str,
                    help='characteristic name for the experiment - to be used for log files, checkpoints, etc (it should be a single word)') 
parser.add_argument('--pic_dir', default='/mnt/data/jschne02/Data/Nihal/3DayData', type=str,
                    help='the base image folder having week-1,2,3 directories')
parser.add_argument('--replicate', default='1', type=str,
                    help='which replicate? - 1,2 or 3')
parser.add_argument('--num_outputs', default=20, type=int,
                    help='number of output classes (by default we have 20 outputs for each of 20 flies)')
parser.add_argument('-d','--data_aug', default='normal', type=str,
                    help='options: r_mask (default: normal)')
parser.add_argument('--lr_scheme', default='normal', type=str, 
                    help='normal: imagenet scheme, ganin: used in DANN paper by Ganin et al (default:imagenet scheme)')
parser.add_argument('--log_path', default='/mnt/data/jschne02/Data/Nihal/logs', type=str,
                    help='path to store terminal output')
parser.add_argument('--chkpt_freq', default=10, type=int,
                    help='frequency at which checkpoints are saved')
parser.add_argument('--chkpoint_path', default='/mnt/data/jschne02/Data/Nihal/checkpoints', type=str,
                    help='path to store checkpoints')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')


global args
args = parser.parse_args()


def main():

    global args, best_prec1, best_domain_acc, log_path, chkpoint_path, mu, sigma, fptr 

    # USER SPECIFICATIONS:
    # ^^^^^^^^^^^^^^^^^^^^

    # parameters:
    mu = 0.73
    sigma = 0.18

    # Data loading code
    source_dir = args.pic_dir + '/week' + args.replicate + '/day1_seq_splits/train'
    targ_dir = args.pic_dir + '/week' + args.replicate + '/day2_seq_splits/train'
    val_source_dir = args.pic_dir + '/week' + args.replicate + '/day1_seq_splits/validate'
    val_targ_dir = args.pic_dir + '/week' + args.replicate + '/day2_seq_splits/validate'
    
    log_path = args.log_path
    chkpoint_path = args.chkpoint_path
    normalize = transforms.Normalize(mean=[mu,mu,mu],
                                     std=[sigma,sigma,sigma])

    # transformations on images
    if args.data_aug=='r_mask':
        trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            HolePunch(n_holes=1, radius=56, square=False),
        ])
    elif args.data_aug=='normal':
        trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        print('Wrong Data Augmentation Input!')

    # Data loader for source directory (training data)
    train_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(source_dir, trans),
        batch_size=args.batch_size//2, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # Data loader for target directory (training data)
    train_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(targ_dir, trans),
        batch_size=args.batch_size//2, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # Data loader for source directory (valdiation data)
    val_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_source_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size//2, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Data loader for target directory (validation data)
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_targ_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size//2, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model (the feat_extractor and the c_classifier together form the resnet18)
    # feature extractor model
    feat_extractor = ResNet(BasicBlock, [2, 2, 2])
    # class classifier model
    c_classifier = class_classifier(BasicBlock, [2], args.num_outputs)
    # domain classifier model
    d_classifier = domain_classifier()


    feat_extractor = torch.nn.DataParallel(feat_extractor).cuda()
    c_classifier = torch.nn.DataParallel(c_classifier).cuda()
    d_classifier = torch.nn.DataParallel(d_classifier).cuda()

    # define optimizer
    fe_optimizer = torch.optim.SGD(feat_extractor.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

    cc_optimizer = torch.optim.SGD(c_classifier.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

    dc_optimizer = torch.optim.SGD(d_classifier.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)    

    class_crit = nn.CrossEntropyLoss().cuda()
    domain_crit = nn.BCELoss().cuda()

    # END OF USER SPECS
    # ^^^^^^^^^^^^^^^^^


    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader1, val_loader2, feat_extractor, c_classifier, d_classifier, class_crit, domain_crit)
        return

    fptr = open(log_path + '/' + args.descriptor, 'a+')

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(fe_optimizer, cc_optimizer, dc_optimizer, epoch)

        # train for one epoch
        train(train_loader1, train_loader2, feat_extractor, c_classifier, d_classifier, class_crit, 
            domain_crit, fe_optimizer, cc_optimizer, dc_optimizer, epoch)

        # evaluate on validation set
        valLoss, val_class_acc, val_domain_acc, cmat = validate(val_loader1, val_loader2, feat_extractor, c_classifier, 
            d_classifier, class_crit, domain_crit)


        # remember best prec@1 and save checkpoint
        is_best = val_class_acc > best_prec1

        if is_best:
            best_cmat = cmat 

        is_best2 = val_domain_acc > best_domain_acc
        best_prec1 = max(val_class_acc, best_prec1)
        best_domain_acc = max(val_domain_acc, best_domain_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_fe': feat_extractor.state_dict(),
            'state_dict_cc': c_classifier.state_dict(),
            'state_dict_dc': d_classifier.state_dict(),
            'class_acc': val_class_acc,
            'domain_acc': val_domain_acc,
            'optimizer_fe' : fe_optimizer.state_dict(),
            'optimizer_cc' : cc_optimizer.state_dict(),
            'optimizer_dc' : dc_optimizer.state_dict(),
        }, [is_best, is_best2], args.chkpt_freq)
        print(' * Best Classification Accuracy: %.3f, Best Domain Accuracy: %.3f' %(best_prec1,best_domain_acc))
        fptr.write(' * Best Classification Accuracy: %.3f, Best Domain Accuracy: %.3f \n' %(best_prec1,best_domain_acc))
    
    fptr.close()

    best_cmat.print_cmat()
    best_cmat.print_pmat()


def train(train_loader1, train_loader2, feat_extractor, c_classifier, d_classifier, class_crit, domain_crit, fe_optimizer, cc_optimizer, dc_optimizer, epoch):

    global fptr, args 

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    day2_iter = iter(train_loader2)

    # switch to train mode
    feat_extractor.train()
    c_classifier.train()
    d_classifier.train()

    end = time.time()
    for i, (input1, c_target1) in enumerate(train_loader1):
        (input2, c_target2) = next(day2_iter)

        # obtain domain targets (0 for source domain, 1 for targ domain)
        d_target1 = c_target1<-5
        # t_target is supposed to be all ones (for target domain)
        d_target2 = c_target2>-5

        # concatenate inputs and targets
        input = torch.cat([input1,input2],0)
        c_target = torch.cat([c_target1,c_target2],0)
        d_target = torch.cat([d_target1,d_target2],0)

        # shuffle the above input and targets
        inds = torch.randperm(input.size()[0])
        input = shuffle_input(input, inds)
        [c_target, d_target] = shuffle_targets([c_target, d_target], inds)
        # place targets on GPU
        c_target = c_target.cuda(async=True)
        d_target = d_target.cuda(async=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        c_target_var = torch.autograd.Variable(c_target)
        d_target_var = torch.autograd.Variable(d_target)

        # compute output
        features = feat_extractor(input_var)
        class_output = c_classifier(features)
        domain_output = d_classifier(features)
        net_loss = class_crit(class_output, c_target_var) + domain_crit(domain_output, d_target_var.float())

        # measure accuracy and record loss
        prec1, prec3 = accuracy(class_output.data, c_target, topk=(1, 3))
        losses.update(net_loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        fe_optimizer.zero_grad()
        cc_optimizer.zero_grad()
        dc_optimizer.zero_grad()
        net_loss.backward()
        dc_optimizer.step()        
        cc_optimizer.step()
        fe_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader1), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))
            fptr.write('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\n'.format(
                   epoch, i, len(train_loader1), loss=losses, top1=top1, top3=top3))


def validate(val_loader1, val_loader2, feat_extractor, c_classifier, d_classifier, class_crit, domain_crit):

    global fptr, args 

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    domain_acc = AverageMeter()
    cmat = confusion_matrix(args.num_outputs)

    day2_iter = iter(val_loader2)

    # switch to evaluate mode
    feat_extractor.eval()
    c_classifier.eval()
    d_classifier.eval()

    end = time.time()
    for i, (input1, c_target1) in enumerate(val_loader1):

        (input2, c_target2) = next(day2_iter)

        # obtain domain targets (0 for source domain, 1 for targ domain)
        d_target1 = c_target1<-5
        # t_target is supposed to be all ones (for target domain)
        d_target2 = c_target2>-5

        # concatenate inputs and targets
        input = torch.cat([input1,input2],0)
        c_target = torch.cat([c_target1,c_target2],0)
        d_target = torch.cat([d_target1,d_target2],0)
        
        # shuffle the above input and targets
        inds = torch.randperm(input.size()[0])
        input = shuffle_input(input, inds)
        [c_target, d_target] = shuffle_targets([c_target, d_target], inds)

        # place targets on GPU
        c_target = c_target.cuda(async=True)
        d_target = d_target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        c_target_var = torch.autograd.Variable(c_target)
        d_target_var = torch.autograd.Variable(d_target)

        # compute output
        features = feat_extractor(input_var)
        class_output = c_classifier(features)
        domain_output = d_classifier(features)
        net_loss = class_crit(class_output, c_target_var) + domain_crit(domain_output, d_target_var.float())

        # update confusion matrix
        cmat.update(c_target,class_output)

        # measure accuracy and record loss (for class classifier)
        prec1, prec3 = accuracy(class_output.data, c_target, topk=(1, 3))
        losses.update(net_loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # measure domain accuracy
        d_acc = domain_accuracy(domain_output.data, d_target)
        domain_acc.update(d_acc,input.size(0)) 
        losses.update(net_loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader1), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))
            fptr.write('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\n'.format(
                   i, len(val_loader1), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))


    print(' * Class Accuracy: {top1.avg:.3f}, Domain Accuracy: {domain_acc.avg:.3f}'
          .format(top1=top1, domain_acc=domain_acc))
    fptr.write(' * Class Accuracy: {top1.avg:.3f}, Domain Accuracy: {domain_acc.avg:.3f} \n'
          .format(top1=top1, domain_acc=domain_acc))

    return [losses.avg,top1.avg,domain_acc.avg,cmat]

def save_checkpoint(state, flags, freq):
    torch.save(state, chkpoint_path+'/'+args.descriptor+'.pth.tar')
    if state['epoch']%freq == 0:
        torch.save(state, chkpoint_path+'/'+args.descriptor+'_epoch'+str(state['epoch'])+'.pth.tar')
    if flags[0]:
        shutil.copyfile(chkpoint_path+'/'+args.descriptor+'.pth.tar'
                        , chkpoint_path+'/'+args.descriptor+'_model_bestClassAcc.pth.tar')
    if flags[1]:
        shutil.copyfile(chkpoint_path+'/'+args.descriptor+'.pth.tar'
                        , chkpoint_path+'/'+args.descriptor+'_model_bestDomAcc.pth.tar')

class confusion_matrix(object):
    """creates a confusion matrix of the network and updates it"""
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros([num_classes,num_classes])
        self.mat_perc = np.zeros([num_classes,num_classes])
        # the sum of probs are stored in prob_total
        self.prob_total = np.zeros([num_classes,num_classes])
        # using prob_total and mat, prob_avg is calculated
        self.prob_avg = np.zeros([num_classes,num_classes])

    def update(self, target, output):
        # get prediction from output
        _, pred = output.topk(1, 1, True, True)
        pred = pred.data.cpu().numpy()

        # get probabilities from output (output is linear)
        prob_mat = self.get_probs(output)

        # update mat and prob_total
        bs = target.size(0) # batch_size
        for i in range(bs):
            self.mat[target[i],pred[i]] = self.mat[target[i],pred[i]] + 1
            self.prob_total[target[i],pred[i]] = self.prob_total[target[i],pred[i]] + prob_mat[i,pred[i]]

        # update mat_perc and prob_avg
        count = self.mat.sum(1) # stores the row-wise total count
        for row in range(self.num_classes):
            for col in range(self.num_classes):
                if(count[row]!=0):
                    self.mat_perc[row][col] = self.mat[row][col]*100/count[row]
                if(self.mat[row,col]!=0):
                    self.prob_avg[row,col] = self.prob_total[row,col]/self.mat[row,col]


    def get_probs(self, output):
        # we mimic what CrossEntropyLoss does by taking exponentials and normalizing
        output = torch.exp(output)
        prob_mat = output.data.cpu().numpy()  # convert to numpy array
        rows,cols = prob_mat.shape
        for i in range(rows):
            s = prob_mat[i,:].sum()
            prob_mat[i,:] = prob_mat[i,:]/s
        return prob_mat


    def print_cmat(self):
        np.set_printoptions(precision=1)
        np.set_printoptions(suppress=True)
        print(self.mat)
        print(self.mat_perc)

    def print_pmat(self):
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        print(self.prob_avg)

    def save_mats(self):
        mat1 = np.around(self.mat,decimals=1)
        mat2 = np.around(self.mat_perc,decimals=1)
        mat3 = np.around(self.prob_avg,decimals=2)

        # stack the mats in row-wise fashion
        mats = np.row_stack((mat1,mat2,mat3))

        path = csv_path + '/' + args.descriptor + '.csv'
        fptr = open(path,'ab')  # csv is a binary file, hence append in binary mode
        np.savetxt(fptr,mats,delimiter='\t')
        fptr.close()

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


def adjust_learning_rate(optimizer1, optimizer2, optimizer3, epoch):
    if args.lr_scheme == 'normal':
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))        
    elif args.lr_scheme == 'ganin':
        p = epoch/90
        lr = 0.01/((1 + 10*p)**0.75)

    print('learning rate is ' + str(lr))
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer3.param_groups:
        param_group['lr'] = lr

def domain_accuracy(output,target):
    output = output[:,0]
    output = output>=0.5
    dif = output-target
    return ((dif==0).sum()/target.size()[0])*100

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

def shuffle_input(input, inds):
    # input is of type Float Tensor
    new_input = torch.FloatTensor(input.size())
    for i in range(inds.size()[0]):
        new_input[i] = input[inds[i],:,:,:]

    return new_input

def shuffle_targets(targs, inds):
    targ1 = targs[0]
    targ2 = targs[1]
    # targets are of type Long Tensor and Byte Tensor
    new_targ1 = torch.LongTensor(targ1.size())
    new_targ2 = torch.ByteTensor(targ2.size())
    for i in range(inds.size()[0]):
        new_targ1[i] = targ1[inds[i]]
        new_targ2[i] = targ2[inds[i]]

    return [new_targ1,new_targ2]


if __name__ == '__main__':
    main()
