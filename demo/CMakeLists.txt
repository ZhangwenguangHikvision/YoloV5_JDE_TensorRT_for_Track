cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(itest)

# C++标准
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA 10.0 REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/ubuntu/workspace/Gitwork/ZJLAB_ObjectTrack_Deepsort/opencv/")
find_package(OpenCV 3.4.8 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    opencv lib : ${OpenCV_LIBRARIES} ")

set(TensorRT-ROOT /usr/local/TensorRT-7.0.0.11)
set(OpenCV-ROOT /home/ubuntu/workspace/Gitwork/ZJLAB_ObjectTrack_Deepsort/opencv)
set(CUDA-ROOT /usr/local/cuda-10.0)
link_directories(../lib)
link_directories(/usr/local/cuda-10.0/lib64)
link_directories(/usr/local/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdeprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnarrowing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fpermissive -std=gnu99")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdeprecated-declarations")

add_definitions("-Wall -g")
# 指定参与编译的源文件
add_executable(itest src/main.cpp)


# 指定头文件路径
include_directories(../include ${OpenCV_INCLUDE_DIRS})

target_link_libraries(itest ${OpenCV_LIBS}  ${CUDA_LIBRARIES})
target_link_libraries(itest nvinfer)
target_link_libraries(itest libObjectDetectTrack.so)

