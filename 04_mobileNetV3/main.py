#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-08 13:51:45
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.preprocess import load_data
from src.model import MobileNetV3
from src.config import get_args
from src.config import adjust_learning_rate
from src.train_utils import train, validate
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main():
    args = get_args()
    train_loader, test_loader = load_data(args)
    num_classes = 16

    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes, multiplier=args.multiplier, dropout_rate=args.dropout).to(device)
    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    if args.load_pretrained or args.evaluate:
        filename = "best_model_" + str(args.model_mode)
        checkpoint = torch.load("./checkpoint/bg_human.t7")
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc1 = checkpoint['best_acc1']
        acc5 = checkpoint['best_acc5']
        best_acc1 = acc1
        print("Load Model Accuracy1: ", acc1, " acc5: ", acc5, "Load Model end epoch: ", epoch)
    else:
        print("init model load ...")
        epoch = 1
        best_acc1 = 0


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.evaluate:
        acc1, acc5 = validate(test_loader, model, criterion, args)
        print("Acc1: ", acc1, "Acc5: ", acc5)
        return


    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_t = time.time()
    with open("./reporting/"+"best_model_" + args.model_mode+".txt", "w") as f:
        for epoch in range(epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            train(train_loader, model, criterion, optimizer, epoch, args)
            acc1, acc5 = validate(test_loader, model, criterion, args)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                print('Saving ...')
                best_acc5 = acc5
                state = {
                    'model':model.state_dict(),
                    'best_acc1':best_acc1,
                    'best_acc5':best_acc5,
                    'epoch':epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = 'best_model_' + str(args.model_mode)
                torch.save(state, './checkpoint/'+filename+'_ckpt.t7')

            time_interval = time.time() - start_t
            time_split = time.gmtime(time_interval)
            #print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec, end="")
            print(" Test best acc1:", best_acc1, " acc1: ", acc1, " acc5: ", acc5)

            f.write("Epoch: " + str(epoch) + " " + " Best acc: " + str(best_acc1) + " Test acc: " + str(acc1) + "\n")
            f.write("Training time: " + str(time_interval) + " Hour: " + str(time_split.tm_hour) + " Minute: " + str(
                time_split.tm_min) + " Second: " + str(time_split.tm_sec))
            f.write("\n")


if __name__ == '__main__':
    main()
