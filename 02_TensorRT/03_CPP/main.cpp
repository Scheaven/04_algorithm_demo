#include <fstream>
#include <iomanip>
#include <sstream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <map>
#include <chrono>
#include "src/trt_interface.h"
#include "src/img_util.hpp"
#include "src/log4plus_util.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

cv::Mat* deal_dst()
{
    static cv::Mat* dst = NULL;
    return dst;
}

int main(int argc, char const *argv[])
{
    cv::Mat image = cv::imread("/data/disk1/project/data/01_reid/0_1.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "The input image is empty!!! Please check....." << std::endl;
    }
    double total = 0.0;
    double total2 = 0.0;
    double total3 = 0.0;
    // run inference and cout time
    auto t0 = Time::now();
    void* handle  = trtCreate("/data/disk2/tmp/004_algorithm_demo/02_TensorRT/03_CPP/4batch_fp16_True.trt", 1, 128, 256, 3, 2048);

    auto t1 = Time::now();

    float conf[2048*1];

    // DebugP(image.size());
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat dst = cv::Mat::zeros(256, 128, CV_32FC3);
    cv::resize(image, dst, dst.size());
    // cv::imshow("sdf", dst);
    // cv::waitKey(0);

    float* data = img_normal(dst, 1);
    auto t2 = Time::now();

    float prob[2048*1];

    trtDoInfer(handle, data, conf);
    DEBUG("dsfskcxjvsd!");
    DEBUG("我是debug!");
    ERROR("::fo-so4032!");
    ERROR("::ERROR!");


    auto t3 = Time::now();
    fsec fs = t1 - t0;
    fsec f2 = t2 - t1;
    fsec f3 = t3 - t2;
    ms d = std::chrono::duration_cast<ms>(fs);
    ms d2 = std::chrono::duration_cast<ms>(f2);
    ms d3 = std::chrono::duration_cast<ms>(f3);
    total += d.count();
    total2 += d2.count();
    total3 += d3.count();
    // printf("%d\n",d );
    std::cout << "Running time of one image is:" << total << "ms" << total2 << ":" <<total3 << std::endl;
    // for(auto x:conf)
    // {
    //     printf("-- %f\n", x);
    // }

    return 0;
}
