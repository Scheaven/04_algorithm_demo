#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-01 17:30:54
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import numpy as np
from platSMO import PlatSMO
import pickle

class SVMModel:
    """docstring for SVM"""
    def __init__(self, data=[],label=[],C=0,toler=0,maxIter=0,**kernelargs):
        '''
            # 返回原list中每个元素在新的list中对应的索引
            print(np.unique(a,return_inverse=True))
            # (array([1, 2, 3, 4, 5]), array([0, 4, 3, 1, 2, 2, 4]))

            # 返回该元素在list中出现的次数
            print(np.unique(a,return_counts=True))
            # (array([1, 2, 3, 4, 5]), array([1, 1, 2, 1, 2]))

            # 当加参数时，unique()返回的是一个tuple,这里利用了tuple的性质，即有多少个元素即可赋值给对应的多少个变量
        '''
        # print(np.unique(label, return_index=True, return_inverse=True, return_counts = True))
        self.classlabels = np.unique(label)
        self.classNum = len(self.classlabels)
        self.classfyNum = (self.classNum*(self.classNum-1))/2
        self.classfy = []
        self.dataSet = {}
        self.kernelargs = kernelargs
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        for inx in range(np.shape(data)[0]):
            if label[inx] not in self.dataSet.keys():
                self.dataSet[label[inx]] = []
                self.dataSet[label[inx]].append(data[inx][:])
            else:
                self.dataSet[label[inx]].append(data[inx][:])

    # n*(n-1)/2个分类器
    # def train(self):
    #     for class_inx in range(self.classNum):
    #         for overlay_inx in range(class_inx+1, self.classNum):
    #             label = [1]*np.shape(self.dataSet[self.classlabels[class_inx]])[0]
    #             label.extend([-1]*np.shape(self.dataSet[self.classlabels[overlay_inx]])[0])

    #             data = []
    #             data.extend(self.dataSet[self.classlabels[class_inx]])
    #             data.extend(self.dataSet[self.classlabels[overlay_inx]])

    #             # 将label 为 self.classlabels[class_inx] 的数字传入
    #             svm = PlatSMO(np.array(data), np.array(label), self.C, self.toler, self.maxIter, **self.kernelargs)
    #             svm.smoP()
    #             self.classfy.append(svm)
    #     print(len(self.classfy))

    #     self.dataSet = None

    # n-1 个分类器，用1和other的形式
    def train(self):
        for class_inx in range(self.classNum):
            label = [1]*np.shape(self.dataSet[self.classlabels[class_inx]])[0]
            data = []
            data.extend(self.dataSet[self.classlabels[class_inx]])
            for overlay_inx in range(self.classNum):
                if overlay_inx == class_inx:
                    continue
                label.extend([-1]*np.shape(self.dataSet[self.classlabels[overlay_inx]])[0])
                data.extend(self.dataSet[self.classlabels[overlay_inx]])

            # 将label 为 self.classlabels[class_inx] 的数字传入
            svm = PlatSMO(np.array(data), np.array(label), self.C, self.toler, self.maxIter, **self.kernelargs)
            svm.smoP()
            self.classfy.append(svm)
        print(len(self.classfy))


    def save(self, filename):
        writer = open(filename, 'wb')
        pickle.dump(self, writer, 2)
        writer.close()
        print("file writer ok!")

    @staticmethod
    def load(filename):
        reader = open(filename, 'rb')
        svm = pickle.load(reader)
        reader.close()
        print("load svm model")
        return svm

    # 和 n-1 个分类器的train配套
    def evaluation_to_M(self, data, label):
        data_rows = np.shape(data)[0]
        classlabel = []
        count = 0.0
        for n in range(data_rows):
            result = [0]* self.classNum
            index = -1
            for i in range(self.classNum):
                    index += 1
                    s = self.classfy[index]
                    t = s.evalution([data[n]])[0]
                    if t > 0.0:
                        result[i] += 1
            classlabel.append(result.index(max(result)))
            if classlabel[-1] != label[n]:
                count += 1
                print(label[n], classlabel[n])
            else:
                print(label[n], classlabel[n])
        print("error rate:", count/data_rows)

        return classlabel

    # 和 n(n-1)/2 个分类器的train配套
    def evaluation_to_one(self, data, label):
        data_rows = np.shape(data)[0]
        classlabel = []
        count = 0.0
        for n in range(data_rows):
            result = [0]* self.classNum
            index = -1
            for i in range(self.classNum):
                for j in range(i+1, self.classNum):
                    index += 1
                    s = self.classfy[index]
                    t = s.evalution([data[n]])[0]
                    if t > 0.0:
                        result[i] += 1
                    else:
                        result[j] += 1
            classlabel.append(result.index(max(result)))
            if classlabel[-1] != label[n]:
                count += 1
                print(label[n], classlabel[n])
            else:
                print(label[n], classlabel[n])
        print("error rate:", count/data_rows)

        return classlabel


    def predict(self, data):
        result = [0]* self.classNum
        index = -1
        for i in range(self.classNum):
            # for j in range(i+1, self.classNum):
                index += 1
                s = self.classfy[index]
                t = s.predict(data)
                if t > 0.0:
                    result[i] += 1
                # else:
                #     result[j] += 1
        classlabel = result.index(max(result))

        return classlabel
