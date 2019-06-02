"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.train_labels if self.train else self.test_labels)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels

class CIFAR10RandomPixels(datasets.CIFAR10):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomPixels, self).__init__(**kwargs)
    if corrupt_prob > 0:
      self.corrupt_pixels(corrupt_prob)

  def corrupt_pixels(self, corrupt_prob):
    np.random.seed(12345)
    data = np.array(self.train_data if self.train else self.test_data)
    mask = np.random.rand(data.shape[0], data.shape[1], data.shape[2], data.shape[3]) <= corrupt_prob
    rnd_data = np.random.randint(0, 256, data.shape)
    data = (1 - mask) * data + mask * rnd_data
    data = data.astype(np.uint8)
    if self.train:
      self.train_data = data
    else:
      self.test_data = data

class CIFAR10ShufflePixels(datasets.CIFAR10):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10ShufflePixels, self).__init__(**kwargs)
    self.corrupt_pixels(corrupt_prob)

  def corrupt_pixels(self, corrupt_prob):
    np.random.seed(12345)
    data = np.array(self.train_data if self.train else self.test_data)
    data = np.reshape(data, (data.shape[0], -1))
    for i in range(data.shape[0]):
        if np.random.rand() <= corrupt_prob:
            np.random.shuffle(data[i,:])
    data = np.reshape(data, (data.shape[0], 32, 32, 3))
    if self.train:
      self.train_data = data
    else:
      self.test_data = data

class CIFAR10GaussianPixels(datasets.CIFAR10):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10GaussianPixels, self).__init__(**kwargs)
    self.corrupt_pixels(corrupt_prob)

  def corrupt_pixels(self, corrupt_prob):
    np.random.seed(12345)
    data = np.array(self.train_data if self.train else self.test_data)
    for i in range(data.shape[0]):
        if np.random.rand() <= corrupt_prob:
            data[i,:,:,:] = np.random.normal((125.3, 123.0, 113.9), (63.0, 62.1, 66.7), data[i,:,:,:].shape)
    data = data.astype(np.uint8)
    print(data[0,0,:,1])
    print(data[:,:,:,0].mean(), data[:,:,:,1].mean(), data[:,:,:,2].mean())
    if self.train:
      self.train_data = data
    else:
      self.test_data = data

