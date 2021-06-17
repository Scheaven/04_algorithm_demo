#pragma once
// #ifndef TRT_INFER_H
// #define TRT_INFER_H

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
// #include "parserOnnxConfig.h"

#include <map>
#include <chrono>
#include "logging.h"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "log4plus_util.h"
#include <opencv2/opencv.hpp>

#define DebugP(x) std::cout << "Line" << __LINE__ << "  " << #x << "=" << x << std::endl


struct TRT_Infer
{
public:
    samplesCommon::Args gArgs;
    Logger gLogger;

    static IHostMemory* trtModelStream;
    static IRuntime* runtime;
    static ICudaEngine* engine;
    static IExecutionContext* context;

    int INPUT_H = 224;
    int INPUT_W = 224;
    int INPUT_C = 3;
    int OUTPUT_SIZE = 2;
    int BATCHSIZE = 1;
    // char* MODEL_TYPE = "pytorch";

    const char* INPUT_BLOB_NAME = "input_1";
    const char* OUTPUT_BLOB_NAME = "reshape_2";

    const std::string gSampleName = "TensorRT.sample_onnx_image";

public:
    TRT_Infer(std::string model_path, int batchSize, int INPUT_W, int INPUT_H, int INPUT_C, int OUTPUT_SIZE);
    ~TRT_Infer();
    // static TRT_Infer* instance;
    // static TRT_Infer* getInstance(const std::string model_path);
    void doInfer(float* data, float *output);

private:
    void initEngine(const std::string& engineFile, IHostMemory*& trtModelStream);
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
};

// #endif
