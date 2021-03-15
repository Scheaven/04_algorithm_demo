#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-09 18:56:13
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import argparse
import glob
import sys

import cv2
import numpy as np
import tqdm

sys.path.append('../../')
sys.path.append("/data/disk2/opt/05_tensorRT/tiny-tensorrt/build")
import pytrt

def get_parser():
    parser = argparse.ArgumentParser(description="trt model inference")

    parser.add_argument(
        "--model_path",
        default="outputs/trt_model/baseline.engine",
        help="trt model path"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="trt_output",
        help="path to save trt model inference results"
    )
    parser.add_argument(
        "--output_name",
        help="tensorRT model output name"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="width of image"
    )
    return parser


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


if __name__ == "__main__":
    args = get_parser().parse_args()

    trt = pytrt.Trt()

    onnxModel = ""
    engineFile = args.model_path
    customOutput = []
    maxBatchSize = 1
    calibratorData = []
    mode = 2
    trt.CreateEngine(onnxModel, engineFile, customOutput, maxBatchSize, mode, calibratorData)

    if not os.path.exists(args.output): os.makedirs(args.output)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            input_numpy_array = preprocess(path, args.height, args.width)
            trt.DoInference(input_numpy_array)
            feat = trt.GetOutput(args.output_name)
            print('feat::', feat)
