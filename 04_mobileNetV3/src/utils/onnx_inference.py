#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-09 16:43:38
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import argparse
import time
import glob

import cv2
import numpy as np
import onnxruntime
import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="onnx model inference")

    parser.add_argument(
        "--model_path",
        default="onnx_model/baseline.onnx",
        help="onnx model path"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
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
    # original_image = Image.open(image_path)
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img

if __name__ == '__main__':
    args = get_parser().parse_args()

    ort_sess = onnxruntime.InferenceSession(args.model_path)

    input_name = ort_sess.get_inputs()[0].name

    if not os.path.exists(args.output): os.makedirs(args.output)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            image = preprocess(path, args.height, args.width)
            output = ort_sess.run(None, {input_name: image})[0]

            print("softmax:", output)
            # _, pred = output.topk(1,1,True,True)
            # pred = pred.t()

            # print("pred:", pred[0].item())
            t2 = time.time()
            print((t2-t1),"--------end:",1/(t2-t1))



