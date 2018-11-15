'''
Code to train CNNs on fly images 
Single Day Training (SDT) and Double Day Training (DDT) experiments can be performed using this code
Various data augmentation techniques can be used to accompany training
'''

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
from skimage.draw import circle

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

# imports needed for FiveCrop
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numbers
import types
import collections
import warnings


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

'''
BEGIN ROTATION CLASS (used for rotating images)
'''

import scipy.ndimage as ndi

def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image.
    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.
    Arguments
    ---------
    matrix : 3x3 matrix/array
    x : integer
        height dimension of image to be transformed
    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.
    NOTE: this can and certainly should be moved to full torch operations.
    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW

    transform : 3x3 affine transform matrix
        matrix to apply
    """
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
            final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x

class Rotation(object):

    def __init__(self,
                 rotation_range,
                 fill_mode='constant',
                 fill_value=0.,
                 target_fill_mode='nearest',
                 target_fill_value=0.,
                 lazy=False):
        """Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        degree = random.uniform(-self.rotation_range, self.rotation_range)
        theta = math.pi / 180 * degree
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), rotation_matrix,
                fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), rotation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed

'''
END ROTATION CLASS
'''

# HolePunch helps in randomly masking images with holes
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


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--descriptor', default='test', type=str,
                    help='characteristic name for the experiment - to be used for log files, checkpoints, etc (it should be a single word)') 
parser.add_argument('--pic_dir', default='/mnt/data/jschne02/Data/Nihal/3DayData', type=str,
                    help='the base image folder having week-1,2,3 directories')
parser.add_argument('--replicate', default='1', type=str,
                    help='which replicate? - 1,2 or 3')
parser.add_argument('--num_outputs', default=20, type=int,
                    help='number of output classes (by default we have 20 outputs for each of 20 flies)')
parser.add_argument('--ddt', default='false', type=str,
                    help='true: perform double day training (DDT), i.e., train on days-1&2, default: false')
# r_crop: random crop, rs_crop: random-sized cropping, r_crop_mask: random cropping plus masking, 
# r_crop2_rzoom: random cropping (type-2) + random zoom, ...
parser.add_argument('-d','--data_aug', default='normal', type=str,
                    help='options: r_crop, r_crop2, r_crop2_rzoom, rs_crop, r_mask, rotation, r_crop_mask (default: normal)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--log_path', default='/mnt/data/jschne02/Data/Nihal/logs', type=str,
                    help='path to store terminal output')
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

best_prec1 = 0

global args
args = parser.parse_args()


def main():

    global args, best_prec1, log_path, chkpoint_path, mu, sigma, fptr, crop_perc

    # USER SPECIFICATIONS:
    # ^^^^^^^^^^^^^^^^^^^^^^

    if args.ddt=='true':
    	traindir = args.pic_dir + '/week' + args.replicate + '/day12_seq_splits/train'
    	valdir = args.pic_dir + '/week' + args.replicate + '/day12_seq_splits/validate'
    else:
    	traindir = args.pic_dir + '/week' + args.replicate + '/day1_seq_splits/train'
    	valdir = args.pic_dir + '/week' + args.replicate + '/day1_seq_splits/validate'

    log_path = args.log_path
    chkpoint_path = args.chkpoint_path

    # mean and std dev for the images
    mu = 0.73
    sigma = 0.18

    # this will be ignored if data_aug is not r_crop2
    # this is the percentage of the image preserved (new_img_dims = 224*crop_perc)
    crop_perc = 0.7

    # different types of transformations (data augmentation)

    normalize = transforms.Normalize(mean=[mu,mu,mu],
                                     std=[sigma,sigma,sigma])
    normalize2 = transforms.Normalize(mean=[mu,mu,mu,mu,mu],
                                     std=[sigma,sigma,sigma,sigma,sigma])

    # transformations for training data
    if args.data_aug == 'normal' or args.data_aug == 'r_crop2':
        trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.data_aug == 'r_crop2_rzoom':
        # same r_crop2 but with some random zoom (both in and out)
        # each image in minibatch is also differently zoomed
        trans = transforms.Compose([
            transforms.CenterCrop(182),
            transforms.ToTensor(),
        ])
    elif args.data_aug == 'r_crop':
        trans = transforms.Compose([
            transforms.Scale(324),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.data_aug == 'rs_crop':
        trans = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.data_aug == 'r_mask':
        trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            HolePunch(n_holes=1, radius=56, square=False),
        ])
    elif args.data_aug == 'rotation':
        trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            Rotation(180,'nearest'),
        ])
    elif args.data_aug == 'r_crop_mask':
        trans = transforms.Compose([
            transforms.Scale(324),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
            HolePunch(n_holes=1, radius=35, square=False),
        ])
    else:
        print('Wrong Data Augmentation Input!')

    # transformations for validation
    if args.data_aug == 'r_crop2_rzoom':
        trans_val = transforms.Compose([
            transforms.CenterCrop(182),
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.data_aug == 'r_crop' or args.data_aug == 'r_crop_mask':
        trans_val = transforms.Compose([
            transforms.Scale(324),
            FiveCrop(224), # this is a list of PIL Images
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            normalize2,
        ])
    else:
        trans_val = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, trans),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, trans_val),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # END OF USER SPECS
    # ^^^^^^^^^^^^^^^^^^^^


    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
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

    print('data augmentation used is %s ' %(args.data_aug))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    fptr = open(log_path + '/' + args.descriptor, 'a+')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        valLoss, prec1, cmat = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1

        if is_best:
            best_cmat = cmat 

        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        print(' * Best prec@1: %.3f' %(best_prec1))
        fptr.write(' * Best prec@1: %.3f \n' %(best_prec1))

    fptr.close()
    best_cmat.print_cmat()
    best_cmat.print_pmat()


def train(train_loader, model, criterion, optimizer, epoch):

    global crop_perc, fptr

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # if r_crop2_rzoom add in a random zoom
        if args.data_aug == 'r_crop2_rzoom':
            # add a small random zoom
            low = 240
            high = 272

            # pilimage transformation works only on a single image, so we
            # apply transformations image by image
            trans_to_pil = transforms.ToPILImage()
            batch_size = input.size(0)
            # transformations to apply on individual pilimages
            trans = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mu,mu,mu],
                                                 std=[sigma,sigma,sigma]),
            ])
            input2 = torch.FloatTensor(batch_size,3,224,224)
            for k in range(batch_size):
                img_tensor = input[k,:,:,:]
                img_pil = trans_to_pil(img_tensor)

                # generate random no. in b/w low and high
                rand = low + int( random.random()*(high-low+1) )
                scale = transforms.Scale(rand)
                img_pil = scale(img_pil)

                new_img_tensor = trans(img_pil)
                input2[k,:,:,:] = new_img_tensor
            input = input2

        # if r_crop2: show only a square portion of image
        if args.data_aug == 'r_crop2' or args.data_aug == 'r_crop2_rzoom':
            # no. of empty pixels in a dimension
            ep = int( (1-crop_perc)*224 )
            # no. of image pixels in a dimension
            img_dim = 224-ep
            # make 0 to p cols empty
            # make 0 to q rows empty
            p = int(random.random()*ep)
            q = int(random.random()*ep)
            input[:,:,0:q+1,:] = 0
            input[:,:,:,0:p+1] = 0
            # make r to 223 cols empty
            # make s to 223 rows empty
            r = p+img_dim
            s = q+img_dim
            input[:,:,s:224,:] = 0
            input[:,:,:,r:224] = 0

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))
            fptr.write('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\n'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion):

    global fptr, args

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
            fptr.write('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\n'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))


    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))
    fptr.write(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return [losses.avg,top1.avg,cmat]


def save_checkpoint(state, is_best):
    torch.save(state, chkpoint_path+'/'+args.descriptor+'.pth.tar')
    if is_best:
        shutil.copyfile(chkpoint_path+'/'+args.descriptor+'.pth.tar'
                        , chkpoint_path+'/'+args.descriptor+'_model_best.pth.tar')

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
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
