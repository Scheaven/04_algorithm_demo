#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-30 13:58:25
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import torch
import sys
sys.path.append('.')

'''
    python RT文件的生成与推理部分代码
    在jd fastReid上测试 已经没有问题（批次稍微有些出入）
    在mobileNet V3上测试没有问题
'''

import time

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
from PIL import Image
import cv2,os
import torchvision
import numpy as np

max_batch_size = 4  # 明明batch是4 不知为何输出是16的批次
onnx_model_path = "/data/disk1/workspace/06_reid/01_fast_reid/02_fast_reid_inference/fastreid.onnx"
TRT_LOGGER = trt.Logger()

# class HostDeviceMem(object):
#     def init(self, host_mem, device_mem):
#         # """
#         # host_mem: cpu memory
#         # device_mem: gpu memory
#         # """
#         print("-----------11-----------")
#         self.host = host_mem
#         self.device = device_mem

#     def init():
#         # """
#         # host_mem: cpu memory
#         # device_mem: gpu memory
#         # """
#         print("---------22-------------")

#     def __str__(self):
#         return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

#     def __repr__(self):
#         return self.__str__()

def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (256, 128))
    miu = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = np.array(image_cv, dtype=np.float)/255.
    img_np = img_np.transpose((2, 0, 1))
    img_np -= miu
    img_np /= std
    img_np_nchw = img_np[np.newaxis]
    img_np_nchw = np.tile(img_np_nchw,(max_batch_size, 1, 1, 1))
    return img_np_nchw


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        # """
        # host_mem: cpu memory
        # device_mem: gpu memory
        # """
        self.host = host_mem
        self.device = device_mem
        print()

    def __str__(self):
        return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        # print(binding) # 绑定的输入输出
        # print(engine.get_binding_shape(binding)) # get_binding_shape 是变量的大小
        size = trt.volume(engine.get_binding_shape(binding))*engine.max_batch_size
        # volume 计算可迭代变量的空间，指元素个数
        # size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # get_binding_dtype 获得binding的数据类型
        # nptype等价于numpy中的dtype，即数据类型
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes) # cuda分配空间
        # print(int(device_mem)) # binding在计算图中的缓冲地址
        bindings.append(int(device_mem))
        #append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",fp16_mode=False, save_engine=True):
    # """
    # params max_batch_size: 预先指定大小好分配显存
    # params onnx_file_path: onnx文件路径
    # params engine_file_path: 待保存的序列化的引擎文件路径
    # params fp16_mode: 是否采用FP16
    # params save_engine: 是否保存引擎
    # returns: ICudaEngine
    # """
    # 如果已经存在序列化之后的引擎，则直接反序列化得到cudaEngine
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read()) # 反序列化
    else: # 由onnx创建cudaEngine

        # 使用logger创建一个builder
        # builder创建一个计算图 INetworkDefinition
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network,  \
            trt.OnnxParser(network, TRT_LOGGER) as parser: # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
            builder.max_workspace_size = 1<<30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
            builder.max_batch_size = max_batch_size # 执行时最大可以使用的batchsize
            builder.fp16_mode = fp16_mode

            # 解析onnx文件，填充计算图
            if not os.path.exists(onnx_file_path):
                quit("ONNX file {} not found!".format(onnx_file_path))
            print('loading onnx file from path {} ...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model: # 二值化的网络结果和参数
                print("Begining onnx file parsing")
                parser.parse(model.read())  # 解析onnx文件
            #parser.parse_from_file(onnx_file_path) # parser还有一个从文件解析onnx的方法

            print("Completed parsing of onnx file")
            # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
            print("Building an engine from file{}' this may take a while...".format(onnx_file_path))

            #################
            print(network.get_layer(network.num_layers-1).get_output(0).shape)
            # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
            engine=builder.build_cuda_engine(network)  # 注意，这里的network是INetworkDefinition类型，即填充后的计算图
            print("Completed creating Engine")
            if save_engine:  #保存engine供以后直接反序列化使用
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  # 序列化
            return engine


# def allocate_buffers(engine):
#     inputs, outputs, bindings = [], [], []
#     stream = cuda.Stream()
#     for binding in engine:
#         # print(binding) # 绑定的输入输出
#         # print(engine.get_binding_shape(binding)) # get_binding_shape 是变量的大小
#         size = trt.volume(engine.get_binding_shape(binding))*engine.max_batch_size
#         # volume 计算可迭代变量的空间，指元素个数
#         # size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         # get_binding_dtype 获得binding的数据类型
#         # nptype等价于numpy中的dtype，即数据类型
#         # allocate host and device buffers
#         host_mem = cuda.pagelocked_empty(size, dtype) # 创建锁业内存
#         device_mem = cuda.mem_alloc(host_mem.nbytes) # cuda分配空间
#         # print(int(device_mem)) # binding在计算图中的缓冲地址
#         bindings.append(int(device_mem))
#         #append to the appropriate list
#         if engine.binding_is_input(binding):
#             # inputs.append(HostDeviceMem(host_mem, device_mem))
#             inputs.append(HostDeviceMem())
#         else:
#             outputs.append(HostDeviceMem())
#             # outputs.append(HostDeviceMem(host_mem, device_mem))
#     return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # htod： host to device 将数据由cpu复制到gpu device
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 当创建network时显式指定了batchsize， 则使用execute_async_v2, 否则使用execute_async
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

if __name__ == '__main__':
    img_np_nchw = get_img_np_nchw("/data/disk1/project/data/01_reid/0_1.png").astype(np.float32)
    fp16_mode = True
    trt_engine_path ="./3test_fp16_{}.trt".format(fp16_mode)

    engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode) # Build an cudaEngine

    context = engine.create_execution_context() # 创建CudaEngine之后,需要将该引擎应用到不同的卡上配置执行环境
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

    shape_of_output = (max_batch_size, 2048)

    inputs[0].host = img_np_nchw.reshape(-1)

    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size = max_batch_size)
    t2 = time.time()
    print(trt_outputs, trt_outputs[0].shape)

    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    print('TensorRT ok')
    print("Inference time with the TensorRT engine: {}".format(t2-t1))



