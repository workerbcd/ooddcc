import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import numpy as np
import os, sys
from math import floor
import random

class Cifar10(CIFAR10):
    def __init__(self, root: str, train: bool = True, transform= None, target_transform= None, download: bool = False, sample_num=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.sample_num = sample_num
        self.cls_num = 10
        if sample_num is not None:
            self._balance_sample(num_per_class=floor(self.sample_num / self.cls_num), random=True)
    def _balance_sample(self, num_per_class, random=True):
        """Sample a balance number of data per class
        It delves into the pytorch ImageFolder class and modify the following members:
            - samples: a list of tuple of (img_file, num_label)
            - imgs: same as samples
            - targets: a list of num_label corr to samples.
        """
        counter_cls = np.array([0] * self.cls_num)
        num_data = len(self.data)
        go_through_order = np.arange(num_data)

        if random:
            np.random.shuffle(go_through_order)

        # go through data
        img_files_new = []
        labels_new = []
        for idx in go_through_order:
            img_file, num_label = self.data[idx], self.targets[idx]
            if counter_cls[num_label] < num_per_class:
                img_files_new.append(img_file)
                labels_new.append(num_label)
                counter_cls[num_label] += 1

            # stop if the criteria is met
            if np.all(counter_cls >= num_per_class):
                break
        # print(labels_new)
        # print(img_files_new)

        # order the data
        # samples_new = [(img_file, num_label) for num_label, img_file in sorted(zip(labels_new, img_files_new))]
        imgs_new = img_files_new
        targets_new = labels_new

        # update the dataset
        # self.dataset.samples = samples_new
        self.data = imgs_new
        self.targets = targets_new


class Cifar100(CIFAR100):
    def __init__(self, root: str, train: bool = True, transform= None, target_transform= None, download: bool = False, sample_num=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.sample_num= sample_num
        self.cls_num=100
        if sample_num is not None:
            self._balance_sample(num_per_class=floor(self.sample_num / self.cls_num), random=True)
    def _balance_sample(self, num_per_class, random=True):
        """Sample a balance number of data per class
        It delves into the pytorch ImageFolder class and modify the following members:
            - samples: a list of tuple of (img_file, num_label)
            - imgs: same as samples
            - targets: a list of num_label corr to samples.
        """
        counter_cls = np.array([0] * self.cls_num)
        num_data = len(self.data)
        go_through_order = np.arange(num_data)

        if random:
            np.random.shuffle(go_through_order)

        # go through data
        img_files_new = []
        labels_new = []
        for idx in go_through_order:
            img_file, num_label = self.data[idx], self.targets[idx]
            if counter_cls[num_label] < num_per_class:
                img_files_new.append(img_file)
                labels_new.append(num_label)
                counter_cls[num_label] += 1

            # stop if the criteria is met
            if np.all(counter_cls >= num_per_class):
                break
        # print(labels_new)
        # print(img_files_new)

        # order the data
        # samples_new = [(img_file, num_label) for num_label, img_file in sorted(zip(labels_new, img_files_new))]
        imgs_new = img_files_new
        targets_new = labels_new

        # update the dataset
        # self.dataset.samples = samples_new
        self.data = imgs_new
        self.targets = targets_new
