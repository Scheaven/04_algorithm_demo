#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-01 19:31:49
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import numpy as np
import sys

'''
    该文件也是采用SMO进行优化，在选择优化变量时，选择误差步长最大的两个变量进行优化，可以大幅提高优化速度。
    该文件中还加入了核函数（线性核函数，RBF核函数），具体实现参见 kernelTrans(self,x,z)
'''
class PlatSMO:
    def __init__(self, dataMat, classlabels, C,toler,maxIter,**kernelargs):
        self.X = np.array(dataMat)
        self.label = np.array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.rows, self.cols = np.shape(dataMat)
        self.alpha = np.array(np.zeros(self.rows), dtype = 'float64')
        self.b = 0.0
        self.eCache = np.array(np.zeros((self.rows, 2))) # 存储alpha[i]的误差值
        self.K = np.zeros((self.rows, self.rows), dtype='float64')  # 核函数信息
        self.kwargs = kernelargs
        self.SV = ()
        self.SVIndex = None
        for inx_i in range(self.rows):
            for inx_j in range(self.rows):
                self.K[inx_i, inx_j] = self.kernelTrans(self.X[inx_i,:], self.X[inx_j,:])

    '''
        线性核函数，RBF核函数
    '''
    def kernelTrans(self, x, z):
        # print(list(x))
        # print(list(z))
        if np.array(x).ndim != 1 or np.array(z).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return np.sum(x*z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return np.exp(np.sum((x-z)*(x-z))/(-1*theta**2))

    # 计算实例xk的误差
    def  calcEK(self, k):
        fxk = np.dot(self.alpha*self.label, self.K[:,k])+self.b    # 计算xk的预测值 其中k为下标
        Ek = fxk -float(self.label[k])
        print("Ek:", Ek)
        return Ek

    # 更新存储的alpha[k]误差
    def updateEK(self, k):
        Ei = self.calcEK(k)
        self.eCache[k] = [1, Ei]
        print("self.eCache:", self.eCache)


    def selectJrand(self, i, m):
        j = i
        while j==i:
            j = int(np.random.uniform(0, m))
        return j


    def clipAlpha(self, a_j,H,L):
        if a_j > H:
            a_j = H
        if L > a_j:
            a_j = L
        return a_j


    # 选择与alpha[i]相对应的更新alpha[j]
    def selectJ(self, i, Ei):
        maxE = 0.0
        selectJ = 0
        Ej =0.0
        validECacheList = np.nonzero(self.eCache[:,0])[0]
        print("validECacheList:", validECacheList)

        if len(validECacheList) > 1: # 优先选择已经更新过的，并且和alpha[i]差异大的
            for k in validECacheList:
                if k == i: continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxE: # 迭代选择一个与Ei差异最大的
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            print("one:", selectJ, Ej)
            return selectJ, Ej
        else:
            selectJ = self.selectJrand(i, self.rows)
            Ej = self.calcEK(selectJ)
            print("two:", selectJ, Ej)
            return selectJ,Ej

    # 更新alpha[i] 和 alpha[j]的公式和策略
    def innerL(self, i):
        Ei = self.calcEK(i)
        # 满足alpha[i] 的更新的条件则更新 // 更新后alpha[i] 在[L,H]范围内
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
            self.updateEK(i)
            j,Ej = self.selectJ(i,Ei) # 选择一个和alpha[i] 配对的alpha[j]
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()

            #[L,H]区间的计算公式
            if self.label[i] != self.label[j]:
                L = max(0,self.alpha[j]-self.alpha[i])
                H = min(self.C,self.C + self.alpha[j]-self.alpha[i])
            else:
                L = max(0,self.alpha[j]+self.alpha[i] - self.C)
                H = min(self.C,self.alpha[i]+self.alpha[j])

            if L == H:
                return 0

            eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
            if eta >= 0:
                return 0

            #alpha[j]的更新公式
            self.alpha[j] -= self.label[j]*(Ei-Ej)/eta
            self.alpha[j] = self.clipAlpha(self.alpha[j],H,L) # 确保 alpha[j]在H ,L 之间
            self.updateEK(j)
            if abs(alphaJOld-self.alpha[j]) < 0.00001:
                return 0

            #alpha[i]的更新公式
            self.alpha[i] +=  self.label[i]*self.label[j]*(alphaJOld-self.alpha[j])
            self.updateEK(i)

            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0<self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) /2.0

            return 1
        else:
            return 0

    #  smo操作，逐步更新alpha参数
    def smoP(self):
        iter = 0
        entrySet = True
        alphaPairChanged = 0

        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.rows):
                    alphaPairChanged+=self.innerL(i)
                iter += 1
            else:
                nonBounds = np.nonzero((self.alpha > 0)*(self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged+=self.innerL(i)
                iter+=1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True

        self.SVIndex = np.nonzero(self.alpha)[0]
        self.SV = self.X[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]
        self.X = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None

    def evalution(self, testData):
        test = np.array(testData)
        result = []

        m = np.shape(test)[0]
        for i in range(m):
            tmp = self.b
            for j in range(len(self.SVIndex)):
                tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test[i,:])

            while tmp == 0:
                tmp = random.uniform(-1,1)

            if tmp > 0:
                tmp = 1
            else:
                tmp = -1
            result.append(tmp)

        return result


    def predict(self, data):
        test = np.array(data)

        tmp = self.b
        for j in range(len(self.SVIndex)):
            tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test)

        while tmp == 0:
            tmp = random.uniform(-1,1)

        if tmp > 0:
            tmp = 1
        else:
            tmp = -1

        return tmp
