#include <opencv2/opencv.hpp>
#include <iostream>

#include <unistd.h>

#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
// #include <opencv2/algorithm.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include "bgsubcnt.h"

using namespace cv;
using namespace std;

const string keys =
        "{help h usage ? || print this message}"
        "{file           || use file (default is system camera)}"
        "{type           |CNT| bg subtraction type from - CNT/MOG2/KNN"
#ifdef HAVE_OPENCV_CONTRIB
        "/GMG/MOG"
#endif
        "}"
        "{bg             || calculate also the background}"
        "{nogui          || run without GUI to measure times}";


int main(int argc, char** argv ) {
    VideoCapture cap = VideoCapture("/data/disk1/project/data/02_hand/03_hand.mp4");
    if(!cap.isOpened())
    {
        cout << "aaaa" << endl;
        return -1;
    }

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG> pBackSub = cv::cuda::createBackgroundSubtractorMOG(20);
    Mat img;
    cv::cuda::GpuMat gpu;
    cv::cuda::GpuMat fgMask;
    cv::cuda::GpuMat UIMG;
    cv::cuda::GpuMat UMask;
    cv::cuda::GpuMat UMask2;
    cv::Ptr<cv::cuda::Filter> open_Filter;
    cv::Mat element{1,1};
    element = std::move(cv::getStructuringElement(cv::MORPH_ELLIPSE, {3,3}));
    open_Filter = cv::cuda::createMorphologyFilter(MORPH_OPEN, CV_8UC1, element);

    while(true)
    {
        // cap >> img;
        if (!cap.read(img))
        {
            printf("==================\n");
            break;
            // exit(EXIT_FAILURE);
        }
        /* 原样例部分 start*/

        // gpu.upload(img);
        // pBackSub->apply(gpu, fgMask, 0.2);

        // Mat mask;
        // fgMask.download(mask);

        /* 原样例部分 end*/

        /* 检测和腐蚀膨胀都可以用cuda*/
        cv::Mat fgmask2;
        // pBackSub->apply(ori_frame, fgmask2, 0.2);
        UIMG.upload(img);
        // Mat gray;
        // cvtColor(ori_frame, gray, COLOR_BGR2GRAY);
        pBackSub->apply(UIMG, UMask);
        UMask.download(fgmask2);

        cv::Mat fgmask4;
        cv::morphologyEx(fgmask2, fgmask4, cv::MORPH_OPEN, element);
        // open_Filter->apply(hyperPara->UMask, hyperPara->UMask2); //测试还没有cup运行的快
        // UMask2.download(fgmask4);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        std::vector<std::vector<Point>> slct_contours;
        cv::findContours(fgmask4, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point{}); // 这一步的PGU版本没有测试成功



        // imshow("patata", mask);
        // waitKey(1);
    }

  return 0;
}


