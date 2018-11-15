import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import pdb

# launch ipython debugger on exception
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)



'''
FIVE CROP RELATED FUNCTIONS/CLASSES
(five_crop is needed to take five deterministic crops on the validation and test images
when images are randomly cropped (plus optionally masked))
'''

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def center_crop(img, output_size):
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return crop(img, i, j, th, tw)

def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)

class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

'''
END FIVE CROP RELATED STUFF
'''


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Code to test network')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--pic_dir', default='/mnt/data/jschne02/Data/Nihal/3DayData', type=str,
                    help='the base image folder having week-1,2,3 directories')
parser.add_argument('--num_outputs', default=20, type=int,
                    help='number of output classes (by default we have 20 outputs for each of 20 flies)')
parser.add_argument('--replicate', default='1', type=str,
                    help='which replicate? - 1,2 or 3')
parser.add_argument('--day', default='Day1', type=str,
                    help='which day? - Day1, Day2, Day3')
parser.add_argument('--chkpoint_path', default='/mnt/data/jschne02/Data/Nihal/checkpoints', type=str,
                    help='path to store checkpoints')
parser.add_argument('--chkpoint_name', default='test', type=str,
                    help='mention main name of checkpoint - omit _model_best.pth.tar')
# choose the data augmentation used while training the network, and appropriate methods are used for testing
# if r_crop and r_crop_mask is chosen, five_crop function is called to take 5 deterministic crops on every test image
parser.add_argument('-d','--data_aug', default='normal', type=str,
                    help='options: r_crop, r_crop2, r_crop2_rzoom, rs_crop, r_mask, rotation, r_crop_mask (default: normal)')
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

    print_confmat = True    # print confusion matrix
    print_probmat = True    # print probability matrix

    testdir = args.pic_dir + '/week' + args.replicate + '/' + args.day 
    chkpoint_path = args.chkpoint_path + '/' + args.chkpoint_name + '_model_best.pth.tar'

    mu = 0.73
    sigma = 0.18

    # Data loading code
    normalize = transforms.Normalize(mean=[mu,mu,mu],
                                    std=[sigma,sigma,sigma])
    normalize2 = transforms.Normalize(mean=[mu,mu,mu,mu,mu],
                                     std=[sigma,sigma,sigma,sigma,sigma])
    print('loading test data....')

    # transformations for testing
    if args.data_aug == 'r_crop2_rzoom':
        trans_test = transforms.Compose([
            transforms.CenterCrop(182),
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.data_aug == 'r_crop' or args.data_aug == 'r_crop_mask':
        trans_test = transforms.Compose([
            transforms.Scale(324),
            FiveCrop(224), # this is a list of PIL Images
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            normalize2,
        ])
    else:
        trans_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, trans_test),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # END OF USER SPECS
    # ^^^^^^^^^^^^^^^^^

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        model.fc = nn.Linear(512,args.num_outputs)
        print('Last FC layer changed: ')
        print(model.fc)
    elif args.arch == 'resnet50':
        model.fc = nn.Linear(2048,args.num_outputs)
        print('Last FC layer changed: ')
        print(model.fc)
    elif args.arch == 'vgg19_bn'or args.arch == 'vgg19':
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, args.num_outputs),
        )
        print('Last layer changed: ')
        print(model.classifier)
        args.lr = 0.01
        print('learning rate changed to: %f' %(args.lr))
    elif args.arch == 'densenet161':
        model.classifier = nn.Linear(2208, args.num_outputs)
    elif args.arch == 'squeezenet1_0':
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(512, args.num_outputs, 1),
            nn.ReLU(True),
            nn.AvgPool2d(1),
        )
        print('Last layer changed: ')
        print(model.classifier)
    else:
        print('Last FC layer not changed (1000 outputs exist)')
        print(model.fc)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # load a saved checkpoint
    if os.path.isfile(chkpoint_path):
        print("=> loading checkpoint '{}'".format(chkpoint_path))
        checkpoint = torch.load(chkpoint_path)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print('This model has a top1 accuracy of %f' %(best_prec1))
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(chkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(chkpoint_path))

    cudnn.benchmark = True

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # evaluate on validation set
    valLoss, prec1, cmat = validate(test_loader, model, criterion)
    print('Top1 Accuracy on this dataset: %f' %(prec1))
    print('Validation Loss on this dataset: %f' %(valLoss))


def validate(val_loader, model, criterion):
    global args

    print('Validation has started!')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    cmat = confusion_matrix(args.num_outputs)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
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
