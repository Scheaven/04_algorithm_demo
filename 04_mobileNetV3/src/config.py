#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-08 13:56:22
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--datasets_mode", type=str, default="IMAGENET", help="(example: CIFAR10, CIFAR100, IMAGENET), (default: IMAGENET)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, (default: 100)")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch size, (default, 512)")
    parser.add_argument("--learning_rate", type=float, default=1e-1, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout rate, not implemented yet, (default: 0.8)")
    parser.add_argument('--model_mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")
    parser.add_argument("--load_pretrained", type=bool, default=False, help="(default: False)")
    parser.add_argument('--evaluate', type=bool, default=False, help="Testing time: True, (default: False)")
    parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
    parser.add_argument('--print_interval', type=int, default=5, help="training information and evaluation information output frequency, (default: 5)")
    parser.add_argument('--data', default='/data/disk2/01_dataset/06_abandon')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--distributed', type=bool, default=False)

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def predict_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--img", type=str, default="./data/0.img", help="please input img path!")
    parser.add_argument('--model_mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")

    args = parser.parse_args()
    return args
