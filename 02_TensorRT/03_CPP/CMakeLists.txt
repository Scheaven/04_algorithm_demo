cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -DGPU -DCUDNN -DOPENCV -DS_DEBUG -DS_SERVER -g  -lm -fPIC -W -O0 -pthread -Wl,--no-as-needed")  # 编译C脚本时调用GPU环境信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -march=native -DGPU -DCUDNN -DOPENCV -DS_DEBUG -DS_SERVER -g -std=c++17  -W -O0 -pthread -Wl,--no-as-needed")

project(test_pro)
set(CMAKE_CXX_STANDARD 17)
#add_compile_options(-fPIC -ldl -lavcodec -lm -g -Wall  -Wshadow -Wno-sign-compare -w  -pthread)
set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

file(GLOB_RECURSE CURRENT_HEADERS
             ./src/*.hpp
             ./src/*.h)

file(GLOB SOURCES
    ./*.cpp
    ./src/*.cpp)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${SOURCES})

#cudaS
find_package(CUDA REQUIRED)

#message("-- CUDA version:$(CUDA_VERSION)")

# 设置环境的模型查找，如果CUDA版本大于9,则执行CUDA环境的模型查找
if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected")
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -gencode arch=compute_61,code=sm_61 -std=c++17# 不同GPU有不同的算力指数，可查看算力表
    )
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "--device-debug;-lineinfo")
    #find_package(OpenCV REQUIRED) # 查找系统的默认opencv环境
    #message(${OpenCV_LIBS})
endif()


SOURCE_GROUP("Filter" FILES ${filter_sources})

set(TensorRT_DIR /data/disk2/opt/05_tensorRT/TensorRT-7.0.0.11)
set(OpenCV_DIR /data/disk2/opt/01_opencv/opencv4.5.1/build)
set(LOG_DIR /data/disk2/opt/log4cplus)
FIND_PACKAGE(OpenCV REQUIRED)

link_directories(/usr/local/cuda-10.0/lib64)

set(LIBS
        protobuf
        nvinfer
        nvparsers
        nvinfer_plugin
        nvonnxparser
        cudnn
        cublas
        cudart
        log4cplus
        # cudart
        # gomp
        # nvToolsExt
        # cublas
        # stdc++fs  #设置-lstdc++fs 新链接器选项，防止std等gcc库调用不到
        # curand
)

include_directories("/usr/include/x86_64-linux-gnu/")
link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(${TensorRT_DIR}/include)
include_directories(${TensorRT_DIR}/samples/common)
link_directories(${TensorRT_DIR}/lib)
link_directories(${LOG_DIR}/lib)
include_directories(${LOG_DIR}/include)


include_directories(/usr/local/cuda-10.0/include)
include_directories(${PROJECT_SOURCE_DIR}/src/)

add_executable(${PROJECT_NAME} ${CURRENT_HEADERS} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
