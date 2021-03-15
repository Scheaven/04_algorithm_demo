#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-08 14:05:53
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import random_split

def load_data(args):
    if args.datasets_mode == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data, train=True, download =False, transform=transform_train),
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data,train=False, transform=transform_test), # 數據文件爲壓縮的文件路徑
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )

    elif args.datasets_mode == "IMAGENET":

        # traindir = os.path.join(args.data, 'trainingDigits')
        # valdir = os.path.join(args.data, 'testDigits')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            args.data,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        len_train = int(len(train_dataset)*0.9)
        train_data,valid_data = random_split(train_dataset,[len_train,len(train_dataset)-len_train])

        # Check class labels
        print("---------------",train_dataset.classes)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler
        )

        test_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

    return train_loader, test_loader



