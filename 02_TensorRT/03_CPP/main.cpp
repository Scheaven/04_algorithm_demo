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

int batchsize = 1;
int out_size = 2;

cv::Mat* deal_dst()
{
    static cv::Mat* dst = NULL;
    return dst;
}

int main(int argc, char const *argv[])
{
    cv::Mat image = cv::imread("/data/disk1/project/01_py_project/02_classification/345.png");
    if (image.empty())
    {
        std::cout << "The input image is empty!!! Please check....." << std::endl;
    }

    // run inference and cout time
    auto t0 = Time::now();
    void* handle  = trtCreate("/data/disk1/project/01_py_project/02_classification/01_mobilenetV1(keras)/test.txt", batchsize, 224, 224, 3, out_size);

    auto t1 = Time::now();

    // float conf[out_size*batchsize];

    // DebugP(image.size());
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // cv::Mat dst = cv::Mat::zeros(128, 256, CV_8UC3); // 宽高的位置是正确的
    cv::Mat dst = cv::Mat::zeros(224, 224, CV_32FC3); // 宽高的位置是正确的
    std::cout << dst.size() <<std::endl;
    // cv::Mat dst;
    cv::resize(image, dst, dst.size()); // 直接用 cv::Size(128, 256)的化，参数为h,w
    // cv::imshow("sdf", dst);
    // cv::waitKey(0);

    float* data = img_normal(dst, batchsize, "keras_else");
    // float* tmp_data = new float[dst.rows*dst.cols * 3 * batchsize];


    // while(1)
    // {
        // memcpy(tmp_data, data, dst.rows*dst.cols * 3 * batchsize);
        double total = 0.0;
        double total2 = 0.0;
        double total3 = 0.0;
        auto t2 = Time::now();
        // float prob[out_size*batchsize];
        std::shared_ptr<float> prob = std::shared_ptr<float>(new float[out_size*batchsize], std::default_delete<float[]>());
        // std::shared_ptr<float> prob = std::make_shared<float>(new float[out_size*batchsize], std::default_delete<float[]>());
        trtDoInfer(handle, data, prob.get());


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

        std::cout << *prob<<  "::" << prob.use_count()<<std::endl;
        int kkk = 0;
        for(;kkk<out_size*batchsize;kkk++)
        {
                // printf("---- %f\n", *(prob.get());
            // if(kkk%out_size==0)
                std::cout << prob.get()[kkk]<<std::endl;
        }
    // }

    return 0;
}

