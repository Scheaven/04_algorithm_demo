#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-06-11 19:01:21
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import numpy as np
from PIL import Image
import tensorrt as trt
import labels  # from cityscapes evaluation script
import skimage.transform

'''
分割的样例，和自己测试的王俊那边的keras mobilenet 前期处理有所不同

所以很多脚本和方法可能不能直接运行
'''

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

MEAN = (71.60167789, 82.09696889, 72.30508881)
CLASSES = 20
HEIGHT = 512
WIDTH = 1024

def sub_mean_chw(data):
   data = data.transpose((1, 2, 0))  # CHW -> HWC
   data -= np.array(MEAN)  # Broadcast subtract
   data = data.transpose((2, 0, 1))  # HWC -> CHW
   return data

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

def color_map(output):
   output = output.reshape(CLASSES, HEIGHT, WIDTH)
   out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
   for x in range(WIDTH):
       for y in range(HEIGHT):

           if (np.argmax(output[:, y, x] )== 19):
               out_col[y,x] = (0, 0, 0)
           else:
               out_col[y, x] = labels.id2label[labels.trainId2label[np.argmax(output[:, y, x])].id].color
   return out_col



import engine as eng
import inference as inf
import keras
import tensorrt as trt

input_file_path = ‘munster_000172_000019_leftImg8bit.png’
onnx_file = "semantic.onnx"
serialized_plan_fp32 = "semantic.plan"
HEIGHT = 512
WIDTH = 1024

image = np.asarray(Image.open(input_file_path))
img = rescale_image(image, (512, 1024),order=1)
im = np.array(img, dtype=np.float32, order='C')
im = im.transpose((2, 0, 1))
im = sub_mean_chw(im)

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
out = color_map(out)

colorImage_trt = Image.fromarray(out.astype(np.uint8))
colorImage_trt.save(“trt_output.png”)

semantic_model = keras.models.load_model('/path/to/semantic_segmentation.hdf5')
out_keras= semantic_model.predict(im.reshape(-1, 3, HEIGHT, WIDTH))

out_keras = color_map(out_keras)
colorImage_k = Image.fromarray(out_keras.astype(np.uint8))
colorImage_k.save(“keras_output.png”)



'''
下边是我的keras样例
测试可以正常执行

'''
import keras
import tensorrt as trt
from keras.preprocessing import image
import tensorrt as trt

input_file_path = '/data/disk1/project/01_py_project/02_classification/pos_379.jpg'
# onnx_file = "semantic.onnx"
serialized_plan_fp32 = "/data/disk1/project/01_py_project/02_classification/01_mobilenetV1(keras)/gesture.trt"
HEIGHT = 224
WIDTH = 224

# image = np.asarray(Image.open(input_file_path))
# img = rescale_image(image, (HEIGHT, WIDTH),order=1)
# im = np.array(img, dtype=np.float32, order='C')
# im = im.transpose((2, 0, 1))
# im = sub_mean_chw(im)

img = image.load_img(input_file_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print(x.shape)
x = preprocess_input(x)
im = np.asarray(x)

engine = load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)
out = do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, 224, 224)
print(out)
