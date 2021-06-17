#pragma once
#include <iomanip>
#include <sstream>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

// float* img_normal(cv::Mat img, int BATCHSIZE);

static const float kMean[3] = { 0.485f, 0.456f, 0.406f};
static const float keras_EMean[3] = { 103.939f, 116.779f, 123.68f};
static const float kStdDev[3] = { 0.229f, 0.224f, 0.225f};
static const int map_[7][3] = { {0,0,0} ,
                {128,0,0},
                {0,128,0},
                {0,0,128},
                {128,128,0},
                {128,0,128},
                {0,128,0}};

float* img_normal(cv::Mat img, int BATCHSIZE, char* model_type);
