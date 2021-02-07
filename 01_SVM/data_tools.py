#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-01 17:15:20
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os

def loadData(path, maps = None):
    file_list = os.listdir(path)
    dataset = [] # 存储数据信息
    lables =[]
    for file in file_list:
        lines = open(path+"/"+file).readlines();
        rows = len(lines)
        cols = len(lines[0].strip())
        line= []
        for rows_inx in range(0,rows):
            for cols_inx in range(0,cols):
                line.append(float(lines[rows_inx][cols_inx]))

        dataset.append(line)

        label = file.split("_")[0]
        if maps != None:
            lables.append(float(maps[label]))
        else:
            lables.append(float(label))

    return dataset, lables

def loadImage(path, maps = None):
    lines = open(path).readlines();
    rows = len(lines)
    cols = len(lines[0].strip())
    dataset = []
    for rows_inx in range(0,rows):
        for cols_inx in range(0,cols):
            dataset.append(float(lines[rows_inx][cols_inx]))

    return dataset
