from __future__ import print_function
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import copy
       
torch.manual_seed(0)
sample_num = 1000

label_npy = './labels.npy'
if not os.path.exists(label_npy):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True)
    np.save(label_npy, dataset.train_labels)
labels = np.load(label_npy)
labels = torch.from_numpy(labels).cuda()

sigma = torch.randint(0, 10, (sample_num, labels.shape[0]))
sigma = torch.zeros(sigma.shape[0], sigma.shape[1], 10).scatter_(1, 1,
sigma = (2 * torch.randint(0, 2, (labels.shape[0], sample_num)) - 1).float().cuda()
print(sigma.shape, sigma.mean(), sigma[0,:10])

logits_dir = sys.argv[1]
for f in os.listdir(logits_dir):
    logits = np.load(os.path.join(logits_dir, f))
    logits = torch.from_numpy(logits).cuda()
    loss = F.cross_entropy(logits, labels, reduction='none').unsqueeze(0)
    #print(loss.shape, loss)
    rad_f = torch.mm(loss, sigma) / labels.shape[0]
    rad_f = rad_f.squeeze().detach().cpu().numpy()
    print(f, rad_f.max(), rad_f.min(), rad_f.mean())

