#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 15:50
# @Author  : Scheaven
# @File    :  predictor.py
# @description:
import cv2, io
import torch
import torch.nn.functional as F

from modeling.meta_arch.build import build_model
from utils.checkpoint import Checkpointer

import onnx
import torch
from onnxsim import simplify
from torch.onnx import OperatorExportTypes

class ReID_Model(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, original_image):
        """

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (np.ndarray): normalized feature of the model.
        """
        # # the model expects RGB inputs
        # original_image = original_image[:, :, ::-1]
        # # Apply pre-processing to image.
        # image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # # Make shape with a new batch dimension which is adapted for
        # # network input
        # original_image = torch.as_tensor(original_image.astype("float32"))[None]

        # images.sub_(self.pixel_mean).div_(self.pixel_std)



        predictions = self.predictor(original_image)    # 转化libtorch 需要用到
        return predictions

    def torch2onnx(self):
        predictions = self.predictor.to_onnx()    # 转化libtorch 需要用到

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.cuda()
        self.model.eval()

        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            images = image.cuda()
            print(images.shape,images.type,images)
            self.model.eval()
            traced_script_module = torch.jit.trace(self.model, images)
            predictions = self.model(images)
            ''' 生成CPP模型 '''
            traced_script_module.save("model.pt")
            # Normalize feature to compute cosine distance
            pred_feat = F.normalize(predictions)
            pred_feat = pred_feat.cpu().data
            return pred_feat

    def to_onnx(self):
        inputs = torch.randn(1, 3, self.cfg.INPUT.SIZE_TEST[0], self.cfg.INPUT.SIZE_TEST[1]).cuda()
        onnx_model = self.export_onnx_model(self.model, inputs)

        model_simp, check = simplify(onnx_model)

        model_simp = self.remove_initializer_from_input(model_simp)

        assert check, "Simplified ONNX model could not be validated"

        # PathManager.mkdirs(args.output)

        onnx.save_model(model_simp, f"fastreid.onnx")

        # logger.info(f"Export onnx model in fastreid successfully!")


    def export_onnx_model(self, model, inputs):
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


    def remove_initializer_from_input(self, model):
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


