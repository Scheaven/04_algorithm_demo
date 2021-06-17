#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-06-11 11:37:43
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import keras

from PIL import Image
import skimage.transform
from keras.preprocessing import image


'''
这个样例其实03_main/inference/engine的综合版
是测试王俊 keras mobilenetV3 的分类版本

'''

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

CLASSES = 2
HEIGHT = 224
WIDTH = 224
input_file_path = '/data/disk1/project/01_py_project/02_classification/pos_379.jpg'
serialized_plan_fp32 = "/data/disk1/project/01_py_project/02_classification/01_mobilenetV1(keras)/gesture.trt"

def allocate_buffers(engine, batch_size, data_type):

   # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
   h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
   h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
   # Allocate device memory for inputs and outputs.
   d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

   d_output = cuda.mem_alloc(h_output.nbytes)
   # Create a stream in which to copy inputs/outputs and run inference.
   stream = cuda.Stream()
   return h_input_1, d_input_1, h_output, d_output, stream

def load_images_to_buffer(pics, pagelocked_buffer):
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
   """
   This is the function to run the inference
   Args:
      engine : Path to the TensorRT engine
      pics_1 : Input images to the model.
      h_input_1: Input in the host
      d_input_1: Input in the device
      h_output_1: Output in the host
      d_output_1: Output in the device
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image

   Output:
      The list of output images

   """

   load_images_to_buffer(pics_1, h_input_1)

   with engine.create_execution_context() as context:
       # Transfer input data to the GPU.
       cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

       # Run inference.

       context.profiler = trt.Profiler()
       context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

       # Transfer predictions back from the GPU.
       cuda.memcpy_dtoh_async(h_output, d_output, stream)
       # Synchronize the stream
       stream.synchronize()
       print("-:" , h_output.shape)
       # Return the host output.
       # out = h_output.reshape((batch_size,3, height, width))
       return h_output

def load_engine(trt_runtime, engine_path):
   with open(engine_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

def preprocess_i(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess_test(x):
    x /= 255.
    # x -= 0.5
    # x *= 2.
    return x

def keras_preprocess(x):
  x /=255
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

  x[..., 0] -= mean[0]
  x[..., 1] -= mean[1]
  x[..., 2] -= mean[2]
  if std is not None:
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]
  return x

def tf_preprocess(x):
   x /= 127.5
   x -= 1.
   return x

def else_preprocess(x):
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
      x[..., 0] /= std[0]
      x[..., 1] /= std[1]
      x[..., 2] /= std[2]
    return x

def pytorch_2_keras(x):
  x *= 255
  return x

if __name__ == '__main__':
  img = Image.open(img_path)
  transform = transforms.Compose([
      transforms.Resize([224, 224]),  # [h,w]
      transforms.ToTensor(),
      # transforms.RandomApply(transforms, p=255.)
      ])
  img = transform(img).unsqueeze(0)
  img = img.numpy()
  img = pytorch_2_keras(img)

  img = np.transpose(img,(0,2,3,1))
  # print(img)
  img = else_preprocess(img)


  # img = image.load_img(img_path, target_size=(img_h, img_w))
  # x = image.img_to_array(img)
  # x = np.expand_dims(x, axis=0)
  # print("=",np.asarray(x))

  # x = preprocess_input(x)
  # x = else_preprocess(x)
  # x = keras_preprocess(x)
  # x = preprocess_test(x)
  # img = np.asarray(x)


  engine = load_engine(trt_runtime, serialized_plan_fp32)
  h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)
  out = do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
  print(out)
