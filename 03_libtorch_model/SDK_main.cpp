//
// Created by Scheaven on 2019/11/18.
//
#include "src/remain_filter.h"
#include "time.h"

#include <opencv2/opencv.hpp>           // C++
#include <chrono>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
//    //视频流信息
    VideoCapture cap;
    try {
        cap.open("/data/disk1/project/data/01_reid/top_l_37.avi");
    }catch(exception){
        cout<<"输入视频"<<endl;
        return 0;
    }

    Mat frame, blob;

    while (true) {
        //读取视频帧
        cap >> frame;
        clock_t clock1 = clock();
        auto t1 = std::chrono::steady_clock::now();

        //if(i++%5!= 0)
        //    sleep(1000);

        clock_t t_strat2 = clock();

        if(!frame.empty())
        {
            RemainFilter::getInstance()->whichRemain(frame.data);
        }else{
            cout << "-----------------------over--" << endl;
            break;
        }

        clock_t t_strat3 = clock();
        cout << "rps---"<< (t_strat3 - t_strat2)/1000 <<endl; //cpu 的运行时间

        auto t_len = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
        clock_t clock2 = clock();

        std::cout << "------::-----"<<t_len<<std::endl;
        std::cout << clock1<< "================="<<std::endl;
        // std::
    }
    return 0;
}

