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

float* img_normal(cv::Mat img, int BATCHSIZE);

