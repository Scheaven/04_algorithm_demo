cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fvisibility=hidden -DGPU -DCUDNN -DOPENCV -DS_DEBUG -DS_SERVER -g  -lm -fPIC -W -O0 -pthread -Wl,--no-as-needed")  # 编译C脚本时调用GPU环境信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -DEIGEN_MAX_ALIGN_BYTES=32 -DEIGEN_DONT_VECTORIZE -fPIC -march=native -DGPU -DCUDNN -DOPENCV -DS_DEBUG -DS_SERVER -g -std=c++14  -W -O0 -pthread -Wl,--no-as-needed")

project(s_filter)
set(CMAKE_CXX_STANDARD 14)
#add_compile_options(-fPIC -ldl -lavcodec -lm -g -Wall  -Wshadow -Wno-sign-compare -w  -pthread)
set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

file(GLOB_RECURSE filter_sources
             ./src/*.cpp
             ./src/*.h
)

#cudaS
find_package(CUDA REQUIRED)

#message("-- CUDA version:$(CUDA_VERSION)")

# 设置环境的模型查找，如果CUDA版本大于9,则执行CUDA环境的模型查找
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


SOURCE_GROUP("Filter" FILES ${filter_sources})

set(PYTORCH_DIR /data/disk2/opt/03_libtorch/libtorch)

set(OpenCV_DIR /data/disk2/opt/01_opencv/opencv4.5.1/build)
FIND_PACKAGE(OpenCV REQUIRED)


#check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)
#if(HAVE_OPENCV_CONTRIB)
#    add_definitions(-DHAVE_OPENCV_CONTRIB)
#
#    option(USE_OCV_KCF "Should use the KCF Filter from opencv_contrib for collisions resolving?" ON)
#
#else(HAVE_OPENCV_CONTRIB)
#
#    option(USE_OCV_KCF "Should use the KCF Filter from opencv_contrib for collisions resolving?" OFF)
#
#endif(HAVE_OPENCV_CONTRIB)
#
#option(USE_OCV_UKF "Should use the Unscented Kalman Filter from opencv_contrib?" OFF)
#
#if(USE_OCV_UKF)
#    add_definitions(-DUSE_OCV_UKF)
#else()
#    remove_definitions(-DUSE_OCV_UKF)
#endif(USE_OCV_UKF)
#
#if(USE_OCV_KCF)
#    add_definitions(-DUSE_OCV_KCF)
#else()
#    remove_definitions(-DUSE_OCV_KCF)
#endif(USE_OCV_KCF)


#include_directories(${filter_sources})

#f (CMAKE_COMPILER_IS_GNUCXX)
#   add_library(${PROJECT_NAME} SHARED
#   ${filter_sources}
#   )
#lse(CMAKE_COMPILER_IS_GNUCXX)
#   add_library(${PROJECT_NAME}
#   ${filter_sources}
#   )
#ndif()

link_directories(/usr/local/cuda-10.0/lib64)

set(LIBS
        protobuf
        cudart
        torch
        torch_cuda
        torch_cpu -Wl,--no-as-needed
        c10
        c10_cuda
        gomp
        nvToolsExt
        opencv_core
        opencv_highgui
        #opencv_imgcodecs
        opencv_imgproc
        #opencv_videoio
        opencv_video
        opencv_calib3d
        opencv_features2d
        opencv_flann
        cublas
        stdc++fs  #设置-lstdc++fs 新链接器选项，防止std等gcc库调用不到
        curand
        cuda
        cudnn
        jsoncpp
)



include_directories("/usr/include/x86_64-linux-gnu/")
link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(
        ${PYTORCH_DIR}/include/
        ${PYTORCH_DIR}/include/torch/csrc/api/include
        /usr/local/cuda-10.0/include

)
include_directories(".")
include_directories("./src")
link_directories(${PYTORCH_DIR}/lib)

add_executable(${PROJECT_NAME} SDK_main.cpp ${filter_sources})

target_link_libraries(${PROJECT_NAME} ${LIBS})
