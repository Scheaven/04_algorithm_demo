//
// Created by Scheaven on 2021/5/6.
//
#include "trt_interface.h"
#include "trt_infer.hpp"
#include <iostream>
#include <unistd.h>

void* trtCreate(const char* model_path, int batchSize, int INPUT_W, int INPUT_H, int INPUT_C, int OUTPUT_SIZE)
{
    TRT_Infer *handle = new TRT_Infer(model_path, batchSize, INPUT_W, INPUT_H, INPUT_C, OUTPUT_SIZE);
    // cv::Mat* dst = deal_dst();
    CLog::Initialize("../src/log4cplus.properties");
    // *dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
    // std::cout << *dst << std::endl;
    return (void*)handle;
}

void trtRelease(void* handle);

void* trtDoInfer(void* handle, float* data, float* output)
{
    TRT_Infer *h = (TRT_Infer*) handle;
    // cv::Mat* dst = deal_dst();
    // cv::resize(img, *dst, dst->size());
    // DebugP(dst->size());
    // float* data = normal(dst, BATCHSIZE);
    h->doInfer(data, output);

}

void trtReleaseResult(void* result);
