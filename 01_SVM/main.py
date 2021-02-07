#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-01 16:23:41
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
from data_tools import loadImage
from libSVM_model import SVMModel

if __name__ == "__main__":
    # training
    # dataset, labels = loadImage('/data/disk2/01_dataset/01_digits/trainingDigits')
    # svm_model = SVMModel(dataset, labels, 200, 0.0001, 10000, name='rbf', theta=20)
    # svm_model.train()
    # svm_model.save("s_svm_model.txt")


    #dataset, labels = loadData('/data/disk2/01_dataset/01_digits/testDigits')
    svm_model = SVMModel.load("s_svm_model.txt")
    # svm_model.evaluation_to_M(dataset, labels)



    print("lable:", svm_model.predict(loadImage('/data/disk2/01_dataset/01_digits/testDigits/0_78.txt')))


