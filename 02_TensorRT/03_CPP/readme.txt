---------------------- RT 个人操作版本 -----------------

01_onnx2RT.cpp 包含了onnx向trt的转化以及模型推理部分  对应的image.cpp和image.hpp在src下

src 下的文件是推理部分的代码编译情况 CMakeLists.txt也是给他们用的



重新不熟到3090上的版本bug注意事项
：
TensorRT和1080Ti不一致，使用gLogger时直接使用sample::gLogger(在CPP中使用)
CMakeList.txt 环境也发生变化：
新增
SET(TRT_SAMPLES_SRC ${TensorRT_DIR}/samples)
以及对CPP文件的${TRT_SAMPLES_SRC}/common/logger.*编译

protobuf环境变化，文件要重新生成


环境中的
if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected")
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -gencode arch=compute_61,code=sm_61 -std=c++14# 不同GPU有不同的算力指数，可查看算力表
    )
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "--device-debug;-lineinfo")
    #find_package(OpenCV REQUIRED) # 查找系统的默认opencv环境
    #message(${OpenCV_LIBS})
endif()
似乎可以去掉
