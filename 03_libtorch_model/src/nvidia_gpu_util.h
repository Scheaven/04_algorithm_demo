#ifndef _nvidia_gpu_util_h_
#define _nvidia_gpu_util_h_
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

namespace gpu{
    int getIdleGPU(const int need);
    char *getGpuInfo();
    int nv_get_suitable_gpu(void);
}

#endif
