cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

list(APPEND CMAKE_PREFIX_PATH "/home/ubuntu/workspace/Gitwork/ZJLAB_ObjectTrack_Deepsort/opencv/")
find_package(OpenCV 3.4.8 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    opencv lib : ${OpenCV_LIBRARIES} ")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.0/include)
link_directories(/usr/local/cuda-10.0/lib64)
# tensorrt
include_directories(/usr/local/TensorRT-7.0.0.11/include)
link_directories(/usr/local/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib)

#opencv
include_directories(../../../include ./)
link_directories(/home/ubuntu/workspace/Gitwork/ZJLAB_ObjectTrack_Deepsort/opencv/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

CUDA_ADD_LIBRARY(yolov5_jde SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/common.cpp ${PROJECT_SOURCE_DIR}/yolov5_jde.cpp)
target_link_libraries(yolov5_jde nvinfer cudart)
target_link_libraries(yolov5_jde ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(demo ${PROJECT_SOURCE_DIR}/demo.cpp ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/common.cpp ${PROJECT_SOURCE_DIR}/yolov5_jde.cpp)
target_link_libraries(demo nvinfer)
target_link_libraries(demo cudart)
target_link_libraries(demo ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

