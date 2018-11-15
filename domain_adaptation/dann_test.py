'''
Testing Domain Adversarial Networks on fly images
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

import numpy as np
import math
import os.path
import getpass
import pdb

# launch ipython debugger on exception
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)


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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Code to test network (specify dir in code)')
parser.add_argument('--pic_dir', default='/mnt/data/jschne02/Data/Nihal/3DayData', type=str,
                    help='the base image folder having week-1,2,3 directories')
parser.add_argument('--num_outputs', default=20, type=int,
                    help='number of output classes (by default we have 20 outputs for each of 20 flies)')
parser.add_argument('--replicate', default='1', type=str,
                    help='which replicate? - 1,2 or 3')
parser.add_argument('--day', default='1', type=str,
                    help='replicate has images acquired over 3 consecutive days, select day on which testing is performed? - 1,2 or 3; default:1')
parser.add_argument('--chkpoint_path', default='/mnt/data/jschne02/Data/Nihal/checkpoints', type=str,
                    help='path to store checkpoints')
parser.add_argument('--chkpoint_name', default='test', type=str,
                    help='mention main name of checkpoint - omit _model_bestClassAcc.pth.tar')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

best_prec1 = 0

global args, print_confmat, print_probmat
args = parser.parse_args()



def main():

    global args, mu, sigma, print_confmat, print_probmat

    # USER SPECIFICATIONS:
    # ^^^^^^^^^^^^^^^^^^^^
    print_confmat = True
    print_probmat = True

    testdir = args.pic_dir + '/week' + args.replicate + '/Day' + args.day 
    chkpoint_path = args.chkpoint_path + '/' + args.chkpoint_name + '_model_bestClassAcc.pth.tar'

    mu = 0.73
    sigma = 0.18

    # Data loading code
    normalize = transforms.Normalize(mean=[mu,mu,mu],
                                    std=[sigma,sigma,sigma])
    print('loading test data....')

    # transformations
    trans_test = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # data loader for test set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, trans_test),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # raw model of DANN onto which checkpoint will be loaded
    feat_extractor = ResNet(BasicBlock, [2, 2, 2])
    c_classifier = class_classifier(BasicBlock, [2], args.num_outputs)

    # push the models onto GPU
    feat_extractor = torch.nn.DataParallel(feat_extractor).cuda()
    c_classifier = torch.nn.DataParallel(c_classifier).cuda()

    # END OF USER SPECS
    # ^^^^^^^^^^^^^^^^^


    # load a saved checkpoint
    if os.path.isfile(chkpoint_path):
        print("=> loading checkpoint '{}'".format(chkpoint_path))

        checkpoint = torch.load(chkpoint_path)
        class_acc = checkpoint['class_acc']
        domain_acc = checkpoint['domain_acc']
        epoch = checkpoint['epoch']

        feat_extractor.load_state_dict(checkpoint['state_dict_fe'])
        c_classifier.load_state_dict(checkpoint['state_dict_cc'])

        print('This model has: class accuracy - %.3f, domain accuracy - %.3f' %(class_acc,domain_acc))
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(chkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(chkpoint_path))

    cudnn.benchmark = True

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # evaluate on validation set
    valLoss, prec1, cmat = validate(test_loader, feat_extractor, c_classifier, criterion)
    print('Top1 Accuracy on this dataset: %f' %(prec1))
    print('Validation Loss on this dataset: %f' %(valLoss))


def validate(val_loader, feat_extractor, c_classifier, criterion):

    global args

    print('Validation has started!')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    cmat = confusion_matrix(args.num_outputs)

    # switch to evaluate mode
    feat_extractor.eval()
    c_classifier.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        features = feat_extractor(input_var)
        output = c_classifier(features)
        loss = criterion(output, target_var)

        # update confusion matrix
        cmat.update(target,output)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
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
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))
            if print_confmat:
                print('Confusion Matrix: ')
                cmat.print_cmat()
            if print_probmat:
                print('Avg. Probability Matrix: ')
                cmat.print_pmat()

    return [losses.avg,top1.avg,cmat]

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
