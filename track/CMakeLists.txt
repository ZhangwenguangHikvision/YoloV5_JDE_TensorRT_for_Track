PROJECT(ObjectDetectTrack)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

AUX_SOURCE_DIRECTORY(./src/errmsg DIR_SRCS_errmsg)
AUX_SOURCE_DIRECTORY(./src/thirdPart DIR_SRCS_hungarianoper)
AUX_SOURCE_DIRECTORY(./src/thirdPart/munkres DIR_SRCS_munkres)
AUX_SOURCE_DIRECTORY(./src/thirdPart/munkres/adapters DIR_SRCS_munkres_adapters)
AUX_SOURCE_DIRECTORY(./src/matching DIR_SRCS_matching)
AUX_SOURCE_DIRECTORY(./src/feature DIR_SRCS_feature)
AUX_SOURCE_DIRECTORY(./src/common DIR_SRCS_common)
AUX_SOURCE_DIRECTORY(./src/api DIR_SRCS_api)

list(APPEND CMAKE_PREFIX_PATH "/home/ubuntu/workspace/Gitwork/ZJLAB_ObjectTrack_Deepsort/opencv/")

find_package(OpenCV 3.4.8 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    opencv lib : ${OpenCV_LIBRARIES} ")

find_package(CUDA REQUIRED)

# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# directory of opencv library
link_directories(${OpenCV_LIBRARIES})

# cuda
include_directories(/usr/local/cuda-10.0/include)
link_directories(/usr/local/cuda-10.0/lib64)
# tensorrt
include_directories(/usr/local/TensorRT-7.0.0.11/include)
link_directories(/usr/local/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib)

include_directories(
./src/errmsg
./src/thirdPart
./src/thirdPart/Eigen
./src/thirdPart/munkres
./src/thirdPart/munkres/adapters
./src/matching
./src/feature
./src/common
./src/api
./src/model
../include
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

CUDA_ADD_LIBRARY(ObjectDetectTrack SHARED 
${DIR_SRCS_errmsg} 
${DIR_SRCS_hungarianoper} 
${DIR_SRCS_munkres} 
${DIR_SRCS_munkres_adapters} 
${DIR_SRCS_feature} 
${DIR_SRCS_matching} 
${DIR_SRCS_api}
./src/model/yololayer.cu
./src/model/common.cpp
./src/model/yolov5_jde.cpp
)

target_link_libraries(ObjectDetectTrack nvinfer cudart ${OpenCV_LIBS})

#include(CheckCXXCompilerFlag)  
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)  
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)  
#if(COMPILER_SUPPORTS_CXX11)  
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")  
#elseif(COMPILER_SUPPORTS_CXX0X)  
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  
#else()  
#     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")  
#endif()  
