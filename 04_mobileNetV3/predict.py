#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-02 15:25:05
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import torch
import shutil
import time
import torch.nn as nn
from src.model import MobileNetV3
from src.config import predict_args
from PIL import Image
import torchvision.transforms as transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if __name__ == '__main__':
    args = predict_args()
    num_classes = 2
    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes).to(device)
    filename = "best_model_" + str(args.model_mode)
    checkpoint = torch.load("./checkpoint/40_LARGE_ckpt.t7")
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    acc1 = checkpoint['best_acc1']
    acc5 = checkpoint['best_acc5']
    best_acc1 = acc1
    print("Load Model Accuracy1: ", acc1, " acc5: ", acc5, "Load Model end epoch: ", epoch)


    model.eval()
    with torch.no_grad():
        print("--------start!")
        print(args.img)

        imgs_file = os.listdir(args.img)
        for img_path in imgs_file:
            print(img_path)
            t1 = time.time()
            image = Image.open(args.img+img_path)

            img_transforms = transforms.Compose([
                transforms.Resize([224,224], interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])

            img_torch = img_transforms(image)
            img_torch = torch.unsqueeze(img_torch,0).to(device)
            print("img shape:", img_torch.shape)



            # images = image.cuda()
            # print(images.shape,images.type,images)
            traced_script_module = torch.jit.trace(model, img_torch)
            output = model(img_torch)
            ''' 生成CPP模型 '''
            traced_script_module.save("model.pt")
            # Normalize feature to compute cosine distance

            print("output::", output)
            print("softmax:", output.softmax())
            _, pred = output.topk(1,1,True,True)
            pred = pred.t()

            print("pred:", pred[0].item())
            # if pred.item() != 0:
            #     if not os.path.exists('./0'):
            #         os.makedirs('./0')
            #     # shutil.move(args.img+img_path, os.path.join('0',img_path))
            # t2 = time.time()
            # print((t2-t1),"--------end:",1/(t2-t1))


    # size_test = cfg.INPUT.SIZE_TEST
    # res.append(T.Resize(size_test, interpolation=3))
