#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-08 16:11:38
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import io
import argparse
import sys
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify
from torch.onnx import OperatorExportTypes
from src.model import MobileNetV3
from collections import OrderedDict
from file_utils import PathManager

# logger = setup_logger(name='onnx_export')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_parser():
    parser = argparse.ArgumentParser(description="Convert Pytorch to ONNX model")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='onnx_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--model_mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")
    parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout rate, not implemented yet, (default: 0.8)")
    return parser

def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model

def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    all_passes = onnx.optimizer.get_available_passes()
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnx.optimizer.optimize(onnx_model, passes)
    return onnx_model



if __name__ == '__main__':
    args = get_parser().parse_args()
    model = MobileNetV3(model_mode=args.model_mode, num_classes=16, multiplier=args.multiplier, dropout_rate=args.dropout).to(device)

    checkpoint = torch.load("./checkpoint/40_LARGE_ckpt.t7")
    new_state_dict = OrderedDict()
    for k,v in checkpoint['model'].items():
        name = k[7:]
        # print(name,'-----------',k)
        new_state_dict[name] = v


    model.load_state_dict(new_state_dict)
    epoch = checkpoint['epoch']
    acc1 = checkpoint['best_acc1']
    acc5 = checkpoint['best_acc5']
    best_acc1 = acc1
    print("Load Model Accuracy1: ", acc1, " acc5: ", acc5, "Load Model end epoch: ", epoch)

    model.eval()


    inputs = torch.randn(1,3, 224, 224).to(device)

    print("-----------strt----------")
    onnx_model = export_onnx_model(model, inputs)
    model_simp, check = simplify(onnx_model)
    model_simp = remove_initializer_from_input(model_simp)
    assert check, "Simplified ONNX model could not be validated"
    PathManager.mkdirs(args.output)
    onnx.save_model(model_simp,f'{args.output}/{args.name}.onnx')
    print("--------end-------------")

    # for img_path in imgs_file:
    #     print(img_path)
    #     image = Image.open(args.img+img_path)

    #     img_transforms = transforms.Compose([
    #         transforms.Resize([224,224], interpolation=3),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]),
    #         ])

    #     img_torch = img_transforms(image)
    #     img_torch = torch.unsqueeze(img_torch,0).to(device)
    #     print("img shape:", img_torch.shape)

    #     output = model(img_torch)
    #     print("output::", output)
    #     print("softmax:", output.softmax())
    #     _, pred = output.topk(1,1,True,True)
    #     pred = pred.t()

    #     print("pred:", pred[0].item())
    #     if pred.item() != 0:
    #         if not os.path.exists('./0'):
    #             os.makedirs('./0')
    #         # shutil.move(args.img+img_path, os.path.join('0',img_path))

    #     print("-----------------end-------------------")

