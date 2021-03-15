#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-09 15:32:48
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import argparse
import sys
import numpy as np
import tensorrt as trt


sys.path.append('../../')
from src.utils.file_utils import PathManager

def get_parser():
    parser = argparse.ArgumentParser(description="Convert ONNX to TRT model")

    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='outputs/trt_model',
        help='path to save converted trt model'
    )
    parser.add_argument(
        "--onnx-model",
        default='outputs/onnx_model/baseline.onnx',
        help='path to onnx model'
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

def onnx2trt(
        model,
        save_path,
        log_level='ERROR',
        max_batch_size=1,
        max_workspace_size=1,
        fp16_mode=True,
        strict_type_constraints=True,
        int8_mode=False,
        int8_calibrator=None,
):
    logger = trt.Logger(getattr(trt.Logger,log_level))
    builder = trt.Builder(logger)

    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    with trt.OnnxParser(network, logger) as parser:
        if isinstance(model,str):
            with open(model, 'rb') as f:
                flag = parser.parse(f.read())
        else:
            flag = parser.parse(f.read())
        if not flag:
            for error in range(parser.num_errors):
                print(parser.get_error(error))

        builder.max_batch_size = max_batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size*(1<<25)
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if strict_type_constraints:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        print("Completed parsing of onnx file")
        # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
        print("Building an engine from file {}' this may take a while...".format(model))
        print(network.get_layer(network.num_layers-1).get_output(0).shape)
        engine = builder.build_cuda_engine(network)

        with open(save_path,'wb') as f:
            print("engine.name",engine.name)
            print(engine.serialize())
            f.write(engine.serialize())

        #################

if __name__ == '__main__':
    args = get_parser().parse_args()

    inputs = np.zeros(shape=(1, 224,224,3))
    onnx_model = args.onnx_model
    engineFile = os.path.join(args.output, args.name+'.engine')

    PathManager.mkdirs(args.output)
    onnx2trt(onnx_model, engineFile)


