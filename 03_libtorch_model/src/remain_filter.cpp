#include "remain_filter.h"
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <chrono>
#include "config_util.h"

using namespace std;

string STATICSTRUCT::model_path = "/data/disk1/project/model_dump/06_abandon/model.pt";
RemainFilter* RemainFilter::instance = NULL;

RemainFilter* RemainFilter::getInstance()
{
    if (instance==NULL)
    {
        int gpu_id = gpu::nv_get_suitable_gpu();
        instance = new RemainFilter(gpu_id);
    }
    return instance;
}

RemainFilter::RemainFilter(int gpu_id)
{
    STATICSTRUCT config;
    if (gpu_id==-1)
    {
        cout << gpu_id << "=============="<<endl;
        this->module = torch::jit::load(config.model_path);
        this->module.to(torch::kCPU);
        this->module.eval();
        this->is_gpu = false;
    }else if (torch::cuda::is_available() && torch::cuda::device_count()>= gpu_id)
    {
        this->gpu_id = gpu_id;
        cout << gpu_id << "-=-="<<endl;
        cudaSetDevice(gpu_id);
        this->module = torch::jit::load(config.model_path,torch::Device(torch::DeviceType::CUDA,gpu_id));
        this->module.to(torch::Device(torch::DeviceType::CUDA, gpu_id));
        this->module.eval();
        this->is_gpu = true;
    }
}
RemainFilter::~RemainFilter(){}

int RemainFilter::whichRemain(unsigned char *pBuf)
{
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "do select remain object" << std::endl;
    auto input_tensor = torch::from_blob(pBuf, {1, 224,224,3});
    input_tensor = input_tensor.permute({0,3,1,2});
    input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

    if (is_gpu)
    {
        input_tensor = input_tensor.to(torch::Device(torch::DeviceType::CUDA, this->gpu_id));
    }

    auto out = this->module.forward({input_tensor});
    auto t_len = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
    std::cout << 1000/t_len<< "detetor time:"<<t_len<<std::endl;
    return 0;
}
