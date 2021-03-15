#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-22 11:05:45
# @Author  : Scheaven (snow_mail@foxmail.com)
# @Link    : www.github.com
# @Version : $Id$

import os
import cv2
import imageio
import numpy

# 使用KNN根据前景面积检测运动物体
def bgSubtractorKNN(cap):
    history = 20    # 训练帧数

    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)

    frames = 0
    frame_list=[]
    mask_list=[]
    x_x = 0
    videoWriter2 = cv2.VideoWriter("all.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 8, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
    # videoWriter = cv2.VideoWriter(str(all)+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), 12, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
    while True:
        res, tmp_frame = cap.read()

        # if x_x>2850 and x_x<3000:
        #     # videoWriter = cv2.VideoWriter(str(300)+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
        #     videoWriter.write(tmp_frame)
        if x_x >3000:
            break

        videoWriter2.write(tmp_frame)
        # print(x_x)

        x_x += 1

        if not res:
            break
#         tmp_frame_h,tmp_frame_w,_ = tmp_frame.shape
#         frame = cv2.resize(tmp_frame,(1280,1280*tmp_frame_h//tmp_frame_w))
#         x_y = 0
#         z_x = 0
#         z_y = 0


#         fg_mask = bs.apply(frame)   # 获取 foreground mask

#         if frames < history:
#             frames += 1
#             continue

#         # 对原始帧进行膨胀去噪
#         th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
#         th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
#         dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
#         # 获取所有检测框
#         img, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# #         # draw a bounding box arounded the detected barcode and display the image
# #         cv.drawContours(image, [box], -1, (0, 255, 0), 3)

# #         c = sorted(contours, key=cv.contourArea, reverse=True)[0]
# #         rect = cv.minAreaRect(c)
# #         box = np.int0(cv.boxPoints(rect))

#         for c in contours:
#             # 获取矩形框边界坐标
#             x, y, w, h = cv2.boundingRect(c)
#             # 计算矩形框的面积
#             area = cv2.contourArea(c)
#             if 200 < area:
#                 x_y += 1
#                 z_x = x
#                 z_y = y

#                 # if x_x>13 and x_x<25 and z_x<800 and z_x > 200 and z_y<300:
#                 #     if w > 350:
#                 #         x = z_x + 280
#                 #         w -= 280
#                 #         h -= 80
#                 #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 #     cv2.putText(frame, str(x_x) + " " + str(x_y) + " " + str(z_x+w) + " " + str(z_y),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                 #     print(x_x,x_y,z_x, z_y,w, h)
#                 #     cv2.waitKey(300)

#                 if x_x>24 and x_x<35 and z_x<800 and z_x > 500 and z_y<300:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, str(x_x) + " " + str(x_y) + " " + str(z_x+w) + " " + str(z_y),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     # cv2.waitKey(300)

#                 if x_x>121 and x_x<129 and z_x<510 and z_x > 400 and z_y<220:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, str(x_x) + " " + str(x_y) + " " + str(z_x)+ " " + str(w) + " " + str(z_y),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     # cv2.waitKey(300)


#                 if x_x>108 and x_x<112 and z_x<800 and z_x > 650 and z_y<300:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, str(x_x) + " x " + str(z_x) + " y " + str(z_y) + " " + str(w),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     # cv2.waitKey(500)
#                     # print("========","p:", x_x,"x:",z_x,"y:", z_y,"w:", w,"h:", h)

#                 if x_x>433 and x_x<445 and z_x<800 and z_x > 650 and z_y<300 and h> 100:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, str(x_x) + " x " + str(z_x) + " y " + str(z_y) + " " + str(w),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     # cv2.waitKey(200)
#                     # print("========","p:", x_x,"x:",z_x,"y:", z_y,"w:", w,"h:", h)


#                 if x_x>1379 and x_x<1385 and z_x<950 and z_x > 840 and h > 10 and z_y < 400:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, str(x_x) + " x " + str(z_x) + " y " + str(z_y) + " " + str(w),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     cv2.waitKey(1000)
#                     print("=+++=","p:", x_x,"x:",z_x,"y:", z_y,"w:", w,"h:", h)

#                 if x_x>1384 and x_x<1395 and  z_x<510 and z_x > 400 and z_y<300:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, str(x_x) + " x " + str(z_x) + " y " + str(z_y) + " " + str(w),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     cv2.waitKey(200)
#                     print("========","p:", x_x,"x:",z_x,"y:", z_y,"w:", w,"h:", h)

#                 print(x_x,x_y,z_x, z_y, w, h)
#                 # if x_x>34 and x_x<48 and x_x != 41 and (z_x+w)<800 and z_x > 400 and z_y < 205 and z_y +h <330:
#                 #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 #     cv2.putText(frame, str(x_x) + " " + str(x_y) + " " + str(z_x+w) + " " + str(z_y),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                 #     # print(x_x,x_y,z_x,w, z_y)
#                 #     # cv2.waitKey(50)

#                 # if x_x>30 and x_x<43 and (z_x+w)>800 and z_y >300 and z_y<500:
#                 #     if z_x < 600:
#                 #         x = z_x + w - 120
#                 #         w = w - 143

#                 #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 #     cv2.putText(frame, str(x_x) + " " + str(x_y) + " " + str(z_x+w) + " " + str(z_y),(x + w//2, y + h//2), 2, 0.75, (0,255,255), 2);
#                     # print(x_x,x_y,z_x,w, z_y)
#                     # cv2.waitKey(500)


#         x_x += 1
#         frame_list.append(dilated)
#         mask_list.append(frame[:,:,(2,1,0)])
#         cv2.imshow("frame", frame)
#         cv2.imshow("bgKNN", dilated)
        if cv2.waitKey(1) & 0xff == 27:
            break

    return frame_list, mask_list


if __name__ == '__main__':
    cap = cv2.VideoCapture("/data/disk2/01_dataset/02_shuohuang/04_classification/01_video/173-bai/SSJ30173_20190520_100000.mp4")
    bgKNN = "bgKNN.gif"
    bg_MOG = "bgMOG.gif"
    bg_MOG2 = "bgMOG2.gif"
    bg_GMG = "bgGMG.gif"

    oriKNN = "oriKNN.gif"
    oriMOG = "oriMOG.gif"

    oriMOG2 = "oriMOG2.gif"
    oriGMG = "oriGMG.gif"

    frame_list, mask_list = bgSubtractorKNN(cap)
#     frame_list, mask_list = bgMOG2(cap)
#     frame_list, mask_list = bgMOG(cap)
#     frame_list, mask_list = bgGMG(cap)

#     imageio.mimsave(bg_GMG, mask_list, 'GIF', duration=0.05)
#     imageio.mimsave(oriGMG, frame_list, 'GIF', duration=0.05)
    cap.release()
    cv2.destroyAllWindows()

