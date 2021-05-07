# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:35:24 2020

@author: LWS

An example of convert Pytroch model to onnx.
You should import your model and provide input according your model.
"""
import torch
import sys
sys.path.append('.')

from data.data_utils import read_image
from predictor import ReID_Model
from config import get_cfg
from data.transforms.build import build_transforms
from engine.defaults import default_argument_parser, default_setup
import time

def get_onnx(model, onnx_save_path, example_tensor):

    example_tensor = example_tensor.cuda()

    _ = torch.onnx.export(model,  # model being run
                                  example_tensor,  # model input (or a tuple for multiple inputs)
                                  onnx_save_path,
                                  verbose=False,  # store the trained parameter weights inside the model file
                                  training=False,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output']
                                  )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False

    model = ReID_Model(cfg)  # 特征提取器

    test_transforms = build_transforms(cfg, is_train=False)
    print (args.img_a1)
    img_a1 = read_image(args.img_a1)
    img_a2 = read_image(args.img_a2)
    img_b1 = read_image(args.img_b1)
    img_b2 = read_image(args.img_b2)
    img_a1 = test_transforms(img_a1)
    img_a2 = test_transforms(img_a2)
    img_b1 = test_transforms(img_b1)
    img_b2 = test_transforms(img_b2)

    out = torch.zeros((4, *img_a1.size()), dtype=img_a1.dtype)
    out[0] += img_a1
    out[1] += img_a2
    out[2] += img_b1
    out[3] += img_b2

    # out = torch.unsqueeze(img_a1,0)

    # t1 = time.time()
    # qurey_feat = model.run_on_image(out)
    # t2 = time.time()
    # print("t2-t1:", t2-t1)

    # similarity1 = torch.cosine_similarity(qurey_feat[0], qurey_feat[1], dim=0)
    # t3 = time.time()
    # print("t2-t1:", t3-t2)
    # similarity2 = torch.cosine_similarity(qurey_feat[0], qurey_feat[2], dim=0)
    # similarity3 = torch.cosine_similarity(qurey_feat[0], qurey_feat[3], dim=0)
    # similarity4 = torch.cosine_similarity(qurey_feat[1], qurey_feat[2], dim=0)
    # similarity5 = torch.cosine_similarity(qurey_feat[1], qurey_feat[3], dim=0)
    # similarity6 = torch.cosine_similarity(qurey_feat[2], qurey_feat[3], dim=0)
    # similarity7 = torch.cosine_similarity(qurey_feat[2], qurey_feat[2], dim=0)
    # print(similarity1, similarity2, similarity3, similarity4, similarity5, similarity6, similarity7)
    #
    #

    onnx_save_path = "onnx/resnet50_2.onnx"
    example_tensor = torch.randn(1, 3, 288, 512, device='cuda')

    # 导出模型
    get_onnx(model, onnx_save_path, example_tensor)


