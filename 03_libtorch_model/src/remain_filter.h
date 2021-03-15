#ifndef REMAIN_FILTER_H
#define REMAIN_FILTER_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "nvidia_gpu_util.h"

struct RemainFilter
{
private:
    bool is_gpu;
    torch::jit::script::Module module;
    int gpu_id;

public:
    static RemainFilter* instance;
    static RemainFilter* getInstance();

public:
    RemainFilter();
    ~RemainFilter();
    RemainFilter(int gpu_id);
    int whichRemain(unsigned char *pBuf);

};


#endif
