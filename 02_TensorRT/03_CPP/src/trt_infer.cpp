#include "trt_infer.hpp"
#include "log4plus_util.h"

IHostMemory* TRT_Infer::trtModelStream = NULL;
IRuntime* TRT_Infer::runtime = NULL;
ICudaEngine* TRT_Infer::engine = NULL;
IExecutionContext* TRT_Infer::context = NULL;

TRT_Infer::TRT_Infer(std::string model_path, int batchSize, int INPUT_W, int INPUT_H, int INPUT_C, int OUTPUT_SIZE):
BATCHSIZE(batchSize), INPUT_W(INPUT_W), INPUT_H(INPUT_H), INPUT_C(INPUT_C), OUTPUT_SIZE(OUTPUT_SIZE)
{
    trtModelStream = nullptr;
    initEngine(model_path, this->trtModelStream);
    assert(this->trtModelStream != nullptr);
    DEBUG("Successfully reload trt file!!!!");
}

void TRT_Infer::initEngine(const std::string& engineFile, IHostMemory*& trtModelStream)
{
    /*  loading moding start */
    std::fstream file;
    DEBUG("loading filename from:" + engineFile);
    nvinfer1::IRuntime* trtRuntime;
    //nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
    file.open(engineFile, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    DEBUG("length:"+ to_string(length));
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    // std::cout << "load engine done" << std::endl;
    // std::cout << "deserializing" << std::endl;
    trtRuntime = createInferRuntime(this->gLogger.getTRTLogger());

    //ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
    ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    DEBUG("deserialize done");
    assert(engine != nullptr);
    DEBUG("The engine in TensorRT.cpp is not nullptr");
    trtModelStream = engine->serialize();

    /*  loading moding ends */
    runtime = createInferRuntime(this->gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    assert(context != nullptr);
}

TRT_Infer::~TRT_Infer()
{

}

void TRT_Infer::doInfer(float* data, float *output)
{
    doInference(*this->context, data, output, this->BATCHSIZE);
}


void TRT_Infer::doInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()

    // const int inputIndex = 0;
    // const int outputIndex = 1;
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    DebugP(inputIndex);
    DebugP(outputIndex);
    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize *INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize *INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    // for (int i = 0; i < (INPUT_C * INPUT_H * INPUT_W ); i++)
    // for (int i = batchSize * INPUT_C * INPUT_H * INPUT_W; i > 0; i--)
    // {
    //     std::cout << "::"<< input[i] << std::endl;
    //     // std::cout << "::"<< (float *)buffers[inputIndex] << std::endl;
    //     /* code */
    // }
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // LOG_ERROR(this->gLogger)<< "=====----------======ok" <<std::endl;
}

// float* TRT_Infer::img_preprocess(float* data)
// {
//         //cv::Mat image(INPUT_H, INPUT_W, CV_32FC3);
//     // float * data;
//     // data = (float*)calloc(INPUT_H*INPUT_W * 3 * BATCHSIZE, sizeof(float));

//     DEBUG(" img_preprocess start ");

//     for (int h = 0; h < BATCHSIZE; ++h)
//     {
//         for (int c = 0; c < 3; ++c)
//         {
//             for (int i = 0; i < INPUT_H; ++i)
//             {   //获取第i行首像素指针
//                 // cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
//                 // cv::Vec3b *p2 = image.ptr<cv::Vec3b>(i);
//                 for (int j = 0; j < INPUT_W; ++j)
//                 {
//                     // data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] = (p1[j][c] / 255.0f - kMean[c]) / kStdDev[c];
//                     float tmp_val = data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j];
//                     if (strcmp(MODEL_TYPE, "pytorch")==0)
//                     {
//                         data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] = (tmp_val/255.0f - kMean[c])/kStdDev[c];
//                     }else if(strcmp(MODEL_TYPE, "keras")==0)
//                     {
//                         data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] = (tmp_val/255.0f - kMean[c])/kStdDev[c];
//                     }else if(strcmp(MODEL_TYPE, "keras_else")==0)
//                     {
//                         data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] = (tmp_val*255.0f - keras_EMean[c]);
//                     }else if(strcmp(MODEL_TYPE, "tf")==0)
//                     {
//                         data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] = (tmp_val+1.0f)*127.5;
//                     }else
//                     {
//                         data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] = tmp_val;
//                     }

//                     // printf("==%f\n", data[h * c * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + i * INPUT_W + j] );
//                 }
//             }
//         }
//     }
//     DEBUG(" img_preprocess end ");
//     return data;
// }
