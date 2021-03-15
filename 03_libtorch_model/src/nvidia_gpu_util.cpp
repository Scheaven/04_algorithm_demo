#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>


#include <dlfcn.h>
#include "nvidia_gpu_util.h"
namespace gpu{

#define LOAD_FUNC(l, s) dlsym(l, s)
#define DL_CLOSE_FUNC(l) dlclose(l)

#define CUDAAPI
/**
 * Return values for NVML API calls.
 */
typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,                   //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,       //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,   //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    NVML_ERROR_UNKNOWN = 999            //!< An internal driver error occurred
} nvmlReturn_t;

typedef void * nvmlDevice_t;

/* Memory allocation information for a device. */
typedef struct nvmlMemory_st
{
    unsigned long long total;        //!< Total installed FB memory (in bytes)
    unsigned long long free;         //!< Unallocated FB memory (in bytes)
    unsigned long long used;         //!< Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
} nvmlMemory_t;


/* Information about running compute processes on the GPU */
typedef struct nvmlProcessInfo_st
{
    unsigned int pid;                 //!< Process ID
    unsigned long long usedGpuMemory; //!< Amount of used GPU memory in bytes.
                                      //!< Under WDDM, \ref NVML_VALUE_NOT_AVAILABLE is always reported
                                      //!< because Windows KMD manages all the memory and not the NVIDIA driver
} nvmlProcessInfo_t;

/* Utilization information for a device. */
typedef struct nvmlUtilization_st
{
    unsigned int gpu;                //!< Percent of time over the past second during which one or more kernels was executing on the GPU
    unsigned int memory;             //!< Percent of time over the past second during which global (device) memory was being read or written
} nvmlUtilization_t;

typedef nvmlReturn_t(CUDAAPI *NVMLINIT)(void);  // nvmlInit
typedef nvmlReturn_t(CUDAAPI *NVMLSHUTDOWN)(void);  // nvmlShutdown
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETCOUNT)(unsigned int *deviceCount); // nvmlDeviceGetCount
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETHANDLEBYINDEX)(unsigned int index, nvmlDevice_t *device); // nvmlDeviceGetHandleByIndex
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETDECODERUTILIZATION)(nvmlDevice_t device, unsigned int *utilization,unsigned int *samplingPeriodUs); // nvmlDeviceGetDecoderUtilization
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETENCODERUTILIZATION)(nvmlDevice_t device, unsigned int *utilization,unsigned int *samplingPeriodUs); // nvmlDeviceGetEncoderUtilization
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETMEMORYINFO)(nvmlDevice_t device, nvmlMemory_t *memory); // nvmlDeviceGetMemoryInfo
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETRUNNINGPROCESSES)(nvmlDevice_t device, unsigned int *infoCount,nvmlProcessInfo_t *infos);// nvmlDeviceGetComputeRunningProcesses
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETPPROCESSNAME)(unsigned int pid, char *name, unsigned int length); // nvmlSystemGetProcessName
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETUTILIZATIONRATES)(nvmlDevice_t device, nvmlUtilization_t *utilization); // nvmlDeviceGetUtilizationRates
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETTEMPERATURE)(nvmlDevice_t device, int sensorType, unsigned int *temp); // nvmlDeviceGetTemperature



#define GPU_MAX_SIZE    128


typedef struct nvGpuUnitInfo_st
{
    unsigned int decoder_utilization;
    unsigned int encoder_utilization;
    unsigned int gpu_utilization;
    unsigned int memory_utilization;
    unsigned int temperature;
    unsigned int running_processes;

    unsigned long long memory_total;
    unsigned long long memory_free;
    unsigned long long memory_used;

}nvGpuUnitInfo_t;



typedef struct nvGpuInfo_st
{
    unsigned int device_count;
    nvGpuUnitInfo_t devices[GPU_MAX_SIZE];

}nvGpuInfo_t;


#define RETURN_SUCCESS     0
#define RETURN_ERROR_LOAD_LIB       (-1)
#define RETURN_ERROR_LOAD_FUNC      (-2)
#define RETURN_ERROR_LIB_FUNC       (-3)
#define RETURN_ERROR_NULL_POINTER   (-4)


#define CHECK_LOAD_NVML_FUNC(t, f, s) \
do { \
    (f) = (t)LOAD_FUNC(nvml_lib, s); \
    if (!(f)) { \
        printf("Failed loading %s from NVML library\n", s); \
        retCode = RETURN_ERROR_LOAD_FUNC; \
    } \
} while (0)

static int check_nvml_error(int err, const char *func)
{
    if (err != NVML_SUCCESS) {
        printf(" %s - failed with error code:%d\n", func, err);
        return 0;
    }
    return 1;
}
#define check_nvml_errors(f) \
do{ \
    if (!check_nvml_error(f, #f)) { \
        retCode = RETURN_ERROR_LIB_FUNC; \
    }\
}while(0)



static int get_gpu_info(nvGpuInfo_t *infos)
{

    if(infos == NULL){
        return RETURN_ERROR_NULL_POINTER;
    }

    int retCode = RETURN_SUCCESS;
    void* nvml_lib;
    NVMLINIT                    nvml_init;
    NVMLSHUTDOWN                nvml_shutdown;
    NVMLDEVICEGETCOUNT          nvml_device_get_count;
    NVMLDEVICEGETHANDLEBYINDEX  nvml_device_get_handle_by_index;
    NVMLDEVICEGETDECODERUTILIZATION     nvml_device_get_decoder_utilization;
    NVMLDEVICEGETENCODERUTILIZATION     nvml_device_get_encoder_utilization;
    NVMLDEVICEGETMEMORYINFO     nvml_device_get_memory_info;
    NVMLDEVICEGETRUNNINGPROCESSES       nvml_device_get_running_processes;
    NVMLDEVICEGETPPROCESSNAME   nvml_device_get_process_name;
    NVMLDEVICEGETUTILIZATIONRATES       nvml_device_get_utilization_rates;
    NVMLDEVICEGETTEMPERATURE    nvml_device_get_temperature;

    nvmlDevice_t device_handel;


    unsigned int utilization_value = 0;
    unsigned int utilization_sample = 0;
    int best_gpu = 0;
    unsigned int decoder_used = 100;

    // open the libnvidia-ml.so
    nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);

    if(nvml_lib == NULL){
        return RETURN_ERROR_LOAD_LIB;
    }


    do{
        CHECK_LOAD_NVML_FUNC(NVMLINIT, nvml_init, "nvmlInit");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }
        CHECK_LOAD_NVML_FUNC(NVMLSHUTDOWN, nvml_shutdown, "nvmlShutdown");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETCOUNT, nvml_device_get_count, "nvmlDeviceGetCount");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETHANDLEBYINDEX, nvml_device_get_handle_by_index, "nvmlDeviceGetHandleByIndex");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETDECODERUTILIZATION, nvml_device_get_decoder_utilization, "nvmlDeviceGetDecoderUtilization");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETENCODERUTILIZATION, nvml_device_get_encoder_utilization, "nvmlDeviceGetEncoderUtilization");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETMEMORYINFO, nvml_device_get_memory_info, "nvmlDeviceGetMemoryInfo");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETRUNNINGPROCESSES, nvml_device_get_running_processes, "nvmlDeviceGetComputeRunningProcesses");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETPPROCESSNAME, nvml_device_get_process_name, "nvmlSystemGetProcessName");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETUTILIZATIONRATES, nvml_device_get_utilization_rates, "nvmlDeviceGetUtilizationRates");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETTEMPERATURE, nvml_device_get_temperature, "nvmlDeviceGetTemperature");
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }


        // get gpu info
        check_nvml_errors(nvml_init());\
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        unsigned int device_count = 0;

        check_nvml_errors(nvml_device_get_count(&device_count));
        if(retCode != RETURN_SUCCESS){
            goto gpu_fail;
        }

        infos->device_count = device_count;


        nvmlMemory_t memory_info;
        nvmlUtilization_t gpu_utilization;
        unsigned int process_buf_size = 256;
        nvmlProcessInfo_t process_buf[256];
        char process_name[256];

        memset(process_buf, 0, sizeof(nvmlProcessInfo_t)*100);


        int i = 0;
        for(i = 0; i < device_count; i++){
            check_nvml_errors(nvml_device_get_handle_by_index(i, &device_handel));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            check_nvml_errors(nvml_device_get_decoder_utilization(device_handel, &infos->devices[i].decoder_utilization, &utilization_sample));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            check_nvml_errors(nvml_device_get_encoder_utilization(device_handel, &infos->devices[i].encoder_utilization, &utilization_sample));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            check_nvml_errors(nvml_device_get_memory_info(device_handel, &memory_info));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            infos->devices[i].memory_total = memory_info.total;
            infos->devices[i].memory_free  = memory_info.free;
            infos->devices[i].memory_used  = memory_info.used;

            check_nvml_errors(nvml_device_get_utilization_rates(device_handel, &gpu_utilization));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            infos->devices[i].gpu_utilization = gpu_utilization.gpu;
            infos->devices[i].memory_utilization = gpu_utilization.memory;

            check_nvml_errors(nvml_device_get_temperature(device_handel, 0, &infos->devices[i].temperature));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            // get process info
            process_buf_size = 100;
            memset(process_buf, 0, sizeof(nvmlProcessInfo_t)*100);
            memset(process_name, 0, sizeof(process_name));
            check_nvml_errors(nvml_device_get_running_processes(device_handel, &process_buf_size, process_buf));
            if(retCode != RETURN_SUCCESS){
                goto gpu_fail;
            }

            if(process_buf_size > 0){
                infos->devices[i].running_processes = process_buf_size;
            }

        }
    }while(0);

gpu_fail:
    nvml_shutdown();
    dlclose(nvml_lib);
    return retCode;
}


static void print_gpu_info(nvGpuInfo_t * infos)
{
    printf("device count:%u\n", infos->device_count);

    int i = 0;

    for(i = 0; i < infos->device_count; i++){
        printf("GPU:%d\t, Utilization:[decoder:%u, encoder:%u, gpu:%u, memory:%u], Temperature:%uC, Memory:[total:%llu, free:%llu, used:%llu], process_buf_size:%u\n ",
            i, infos->devices[i].decoder_utilization, infos->devices[i].encoder_utilization, infos->devices[i].gpu_utilization, infos->devices[i].memory_utilization,
            infos->devices[i].temperature, infos->devices[i].memory_total, infos->devices[i].memory_free, infos->devices[i].memory_used, infos->devices[i].running_processes);
    }
}

int nv_get_suitable_gpu_by_mem(nvGpuInfo_t &gpu_info, const int mem){

    int suitable_gpu = -1;
    int mem_idle = mem;
    for(int i = 0; i < gpu_info.device_count; i++){
        int mem_free = gpu_info.devices[i].memory_free >> 20;
        if(mem_free > mem_idle){
            mem_idle = mem_free;
            suitable_gpu = i;
        }
    }

    return suitable_gpu;
}

int nv_get_suitable_gpu(void)
{
    nvGpuInfo_t gpu_info;
    int suitable_gpu = 0; // default gpu is #0
    int i = 0;

    int ret = get_gpu_info(&gpu_info);

    unsigned int min_processes = 2000;
    if(!ret){
        print_gpu_info(&gpu_info);

        for(i = 0; i < gpu_info.device_count; i++){
            //printf("%d\n", i);
            if(gpu_info.devices[i].running_processes < min_processes){
                min_processes = gpu_info.devices[i].running_processes;
                suitable_gpu = i;
            }
        }
    }else{
        return -1;
    }

    return suitable_gpu;
}

int getIdleGPU(const int need){
    nvGpuInfo_t gpu_buf;

    int ret = get_gpu_info(&gpu_buf);

    if(!ret){
        // print_gpu_info(&gpu_buf);
        return nv_get_suitable_gpu_by_mem(gpu_buf, need);
    }

    return -1;
}

    // unsigned int decoder_utilization;
    // unsigned int encoder_utilization;
    // unsigned int gpu_utilization;
    // unsigned int memory_utilization;
    // unsigned int temperature;
    // unsigned int running_processes;

    // unsigned long long memory_total;
    // unsigned long long memory_free;
    // unsigned long long memory_used;

char *getGpuInfo(){

    nvGpuInfo_t gpu_info;
    int flag = get_gpu_info(&gpu_info);

    char * ret= NULL;

    if(!flag){

        char f[] = "gpu";
        char s[] = "|";

        ret = (char*)malloc(1024 * gpu_info.device_count);
        int len = 0;

        char tmp[1024];

        for(int i = 0; i < gpu_info.device_count; i++){
            memset(tmp, 0, 1024);

            nvGpuUnitInfo_t g = gpu_info.devices[i];
            sprintf(tmp, "%s%s%d%s%d%s%d%s%d%s%d%s%llu%s%llu%s%llu%s",
            f,s,
            g.gpu_utilization,s,
            g.temperature,s,
            g.decoder_utilization,s,
            g.encoder_utilization,s,
            g.running_processes,s,
            g.memory_total,s,
            g.memory_free,s,
            g.memory_used,s);

            int l = strlen(tmp);
            memcpy(ret+len, tmp, l);
            len += l;
        }
        ret[len] = '\0';
    }

    return ret;
}

int test(void)
{
    nvGpuInfo_t gpu_buf;

    int ret = get_gpu_info(&gpu_buf);

    if(!ret)
        print_gpu_info(&gpu_buf);

    return nv_get_suitable_gpu();
}



}

//int main(int argc, char const *argv[])
//{
//    /* code */
//    int gpu_device = gpu::nv_get_suitable_gpu();
//    printf("%d\n",gpu_device);
//    return 0;
//}
