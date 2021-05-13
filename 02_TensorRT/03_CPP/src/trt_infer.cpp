#include "trt_infer.hpp"

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
        std::cout << "Successfully reload trt file!!!!" << std::endl;
}

void TRT_Infer::initEngine(const std::string& engineFile, IHostMemory*& trtModelStream)
{
    /*  loading moding start */
    std::fstream file;
    std::cout << "loading filename from:" << engineFile << std::endl;
    nvinfer1::IRuntime* trtRuntime;
    //nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
    file.open(engineFile, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    std::cout << "length:" << length << std::endl;
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    std::cout << "load engine done" << std::endl;
    std::cout << "deserializing" << std::endl;
    trtRuntime = createInferRuntime(this->gLogger.getTRTLogger());
    //ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
    ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    std::cout << "deserialize done" << std::endl;
    assert(engine != nullptr);
    std::cout << "The engine in TensorRT.cpp is not nullptr" <<std::endl;
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

    const int inputIndex = 0;
    const int outputIndex = 1;
    // const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    // const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    DebugP(inputIndex);
    DebugP(outputIndex);
    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    LOG_ERROR(this->gLogger)<< "=====----------======ok" <<std::endl;
}
