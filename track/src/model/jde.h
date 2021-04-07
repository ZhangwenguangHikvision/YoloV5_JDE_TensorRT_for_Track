#ifndef _JDE_
#define _JDE_

#include <vector>
#include <string>
#include "NvInfer.h"
#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "CV_AlgCommon.h"

using namespace nvinfer1;
using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

class JDETrack
{
    public:
		explicit JDETrack();
		~JDETrack();
        int Init(const std::string & config_path, const int & max_batch_size);
        int Run(cv::Mat & input, cv_object_list & output, cv_image * feature_map);
	private:
        JDETrack(const JDETrack &);
        const JDETrack &operator=(const JDETrack &);

        float NMS_THRESH;
        float CONF_THRESH;
        int BATCH_SIZE;

        // stuff we know about the network and the input/output blobs
        int INPUT_H;
        int INPUT_W;
        int CLASS_NUM;
        int OUTPUT_SIZE;  
        int OUTPUT_SIZE2;
        const char* INPUT_BLOB_NAME;
        const char* OUTPUT_BLOB_NAME;
        const char* OUTPUT_BLOB_NAME_2;
        Logger gLogger;
        cudaStream_t stream;

        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;

        void* device_buffers[3];
        float * host_buffers[3];
};


#ifdef __cplusplus
}
#endif

#endif 
