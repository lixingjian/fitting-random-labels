from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import time
import copy
from cifar10_data import CIFAR10RandomLabels,CIFAR10RandomPixels,CIFAR10ShufflePixels,CIFAR10GaussianPixels

import cmd_args
import model_mlp, model_wideresnet, model_inception, model_alexnet

def get_data_loaders(args, shuffle_train=True):
  if args.data == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(28),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.CenterCrop(28),
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        normalize
        ])

    if args.task == 'random_label':
        dataset = CIFAR10RandomLabels
    elif args.task == 'random_pixel':
        dataset = CIFAR10RandomPixels
    elif args.task == 'shuffle_pixel':
        dataset = CIFAR10ShufflePixels
    elif args.task == 'gaussian_pixel':
        dataset = CIFAR10GaussianPixels
    elif args.task == 'normal':
        dataset = torchvision.datasets.CIFAR10
 
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.task == 'normal':
        train_set = dataset(root='./data', train=True, download=True,
                            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            dataset(root='./data', train=False,
                            transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        dump_set = dataset(root='./data', train=True, download=True,
                            transform=transform_train)
        dump_loader = torch.utils.data.DataLoader(dump_set, 
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        train_set = dataset(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob)
        train_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            dataset(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob), 
            batch_size=args.batch_size, shuffle=False, **kwargs)
        dump_set = dataset(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob)
        dump_set.train_labels = copy.deepcopy(train_set.train_labels)
        dump_set.train_data = copy.deepcopy(train_set.train_data) 
        dump_loader = torch.utils.data.DataLoader(dump_set, 
            batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, dump_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(args.data))


def get_model(args):
  # create model
  if args.arch == 'wide-resnet':
    model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate)
  elif args.arch == 'mlp':
    n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
    n_units.append(args.num_classes)  # output dim
    n_units.insert(0, 28*28*3)        # input dim
    model = model_mlp.MLP(n_units)
  elif args.arch == 'inception':
    model = model_inception.inception_v3(num_classes = args.num_classes)
  elif args.arch == 'alexnet':
    model = model_alexnet.alexnet(num_classes = args.num_classes)
  # for training on multiple GPUs.
  # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
  # model = torch.nn.DataParallel(model).cuda()
  model = model.cuda()

  return model


def train_model(args, model, train_loader, val_loader, dump_loader,
                start_epoch=None, epochs=None):
  cudnn.benchmark = True

  # define loss function (criterion) and pptimizer
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  start_epoch = start_epoch or 0
  epochs = epochs or args.epochs

  for epoch in range(start_epoch, epochs):
    since = time.time()
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    tr_loss, tr_prec1 = train_epoch(train_loader, dump_loader, model, criterion, optimizer, epoch, args)
    # evaluate on validation set
    val_loss, val_prec1 = validate_epoch(val_loader, model, criterion, epoch, args)

    if args.eval_full_trainset:
      tr_loss, tr_prec1 = validate_epoch(train_loader, model, criterion, epoch, args)

    time_elapsed = time.time() - since
    logging.info('%03d: Acc-tr: %6.2f, Acc-val: %6.2f, L-tr: %6.4f, L-val: %6.4f, Time: %.0f sec',
                 epoch, tr_prec1, val_prec1, tr_loss, val_loss, time_elapsed)

def dump_epoch(dump_loader, model, criterion, optimizer, epoch, step, args):
  #model.train()
  results = None
  for i, (input, target) in enumerate(dump_loader):
    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    # compute output
    output = model(input_var)
    output = F.softmax(output)
    output = output.detach().cpu().numpy()
    if results is None:
        results = output
    else:
        results = np.concatenate((results, output), axis = 0)
  np.save(('%s/%s_%d_%d.npy' % (args.save, args.task, epoch, step)), results)

def train_epoch(train_loader, dump_loader, model, criterion, optimizer, epoch, args):
  """Train for one epoch on the training set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to train mode
  model.train()

  for i, (input, target) in enumerate(train_loader):
    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if args.save and (i % 10 == 0):
        dump_epoch(dump_loader, model, criterion, optimizer, epoch, i, args)

  return losses.avg, top1.avg


def validate_epoch(val_loader, model, criterion, epoch, args):
  """Perform validation on the validation set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))

  return losses.avg, top1.avg


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


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
  lr = args.learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
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


def setup_logging(args):
  import datetime
  exp_dir = os.path.join('runs', args.exp_name)
  if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
  log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
  logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
  # also log into console
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
  print('Logging into %s...' % exp_dir)


def main():
  args = cmd_args.parse_args()
  if args.steps > 0:
    args.epochs = int(args.steps * args.batch_size / 50000)
  setup_logging(args)
  print(args)

  if args.command == 'train':
    train_loader, val_loader, dump_loader = get_data_loaders(args, shuffle_train=True)
    model = get_model(args)
    logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
    train_model(args, model, train_loader, val_loader, dump_loader)


if __name__ == '__main__':
  main()

