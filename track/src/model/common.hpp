#ifndef JDE_COMMON_H_
#define JDE_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "yololayer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

void jde_cuda_doInference(IExecutionContext& context, cudaStream_t& stream, void **device_buffers, 
    float** host_buffers, int batchSize, const int INPUT_H, const int INPUT_W, const int OUTPUT_SIZE,
    const int OUTPUT_SIZE2) ;
void jde_nms(std::vector<Yolo_Jde::Detection>& res, float *output, float conf_thresh, float nms_thresh);
cv::Rect jde_get_rect(cv::Mat& img, float bbox[4]);
int jde_cuda_preprocess(cv::Mat& img, void* indata, int input_w, int input_h);
#endif

