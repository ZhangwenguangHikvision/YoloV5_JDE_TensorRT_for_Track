#include "jde.h"
#include <sys/time.h>

using namespace nvinfer1;
using namespace cv;
using namespace std;

#define JDE_OUTPUT2_C        76
#define JDE_OUTPUT2_H         136
#define JDE_OUTPUT2_W        512

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000.0 + tv.tv_usec/1000.0;
}

JDETrack::JDETrack()
{
    /*函数初始化*/
    this->NMS_THRESH = 0.4;
    this->CONF_THRESH = 0.5;
    this->BATCH_SIZE = 1;

    // stuff we know about the network and the input/output blobs
    this->INPUT_H = Yolo_Jde::INPUT_H;
    this->INPUT_W = Yolo_Jde::INPUT_W;
    this->CLASS_NUM = Yolo_Jde::CLASS_NUM;
    this->OUTPUT_SIZE = (Yolo_Jde::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo_Jde::Detection) / sizeof(float) + 1);  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    this->OUTPUT_SIZE2 = (JDE_OUTPUT2_C*JDE_OUTPUT2_H*JDE_OUTPUT2_W);
    this->INPUT_BLOB_NAME = "data";
    this->OUTPUT_BLOB_NAME = "prob";
    this->OUTPUT_BLOB_NAME_2 = "embing";
    return;
}

int JDETrack::Init(const std::string & config_path, const int & max_batch_size)
{
    // read config file
    FileStorage fs(config_path, FileStorage::READ);
    string trt_model_Path = fs["trt_path"];

    std::cout<< config_path <<"\n";

    std::ifstream file(trt_model_Path, std::ios::binary);

    size_t size{0};
    char *trtModelStream{nullptr};
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    CHECK(cudaStreamCreate(&this->stream));

    this->runtime = createInferRuntime(this->gLogger);
    CODECVASSERT(this->runtime != nullptr,CODE_CV_ERROR);

    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    CODECVASSERT(this->engine != nullptr,CODE_CV_ERROR);

    this->context = engine->createExecutionContext();
    CODECVASSERT(this->context != nullptr,CODE_CV_ERROR);

    delete[] trtModelStream;

    assert(engine->getNbBindings() == 3);
    

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(this->INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(this->OUTPUT_BLOB_NAME);
    const int outputIndex2 = engine->getBindingIndex(this->OUTPUT_BLOB_NAME_2);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    assert(outputIndex2 == 2);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&(this->device_buffers[inputIndex]), this->BATCH_SIZE * 3 * this->INPUT_H * this->INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(this->device_buffers[outputIndex]), this->BATCH_SIZE * this->OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&(this->device_buffers[outputIndex2]), this->BATCH_SIZE * this->OUTPUT_SIZE2 * sizeof(float)));

    this->host_buffers[inputIndex] = (float *)malloc(this->BATCH_SIZE * 3 * this->INPUT_H * this->INPUT_W * sizeof(float));
    this->host_buffers[outputIndex] = (float *)malloc(this->BATCH_SIZE * this->OUTPUT_SIZE * sizeof(float));
    this->host_buffers[outputIndex2] = (float *)malloc(this->BATCH_SIZE * this->OUTPUT_SIZE2 * sizeof(float));
    return CODE_CV_OK;
}

JDETrack::~JDETrack()
{
    //printf("Destroy JDETrack!\n");
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for(int i = 0; i < 3; i++)
    {
        if(this->device_buffers[i])
        {
            CHECK(cudaFree(this->device_buffers[i]));
            this->device_buffers[i] = NULL;
        }
        if(this->host_buffers[i])
        {
            free(this->host_buffers[i]);
            this->host_buffers[i] = NULL;
        }
    }
    return;
}

int JDETrack::Run(cv::Mat & input, cv_object_list & output, cv_image * feature_map)
{
    double start = get_current_time();
    jde_cuda_preprocess(input, (void*)(this->host_buffers[0]), this->INPUT_W, this->INPUT_H);
    double end = get_current_time();
    printf("jde_cuda_preprocess time cost: %f ms\n", (end - start));
    start = get_current_time();
    jde_cuda_doInference(*this->context, this->stream, this->device_buffers, this->host_buffers, this->BATCH_SIZE,
         this->INPUT_H, this->INPUT_W, this->OUTPUT_SIZE, this->OUTPUT_SIZE2) ;
    end = get_current_time();
    printf("jde_cuda_doInference time cost: %f ms\n", (end - start));
    start = get_current_time();

    std::vector<Yolo_Jde::Detection> res;
    jde_nms(res, this->host_buffers[1], this->CONF_THRESH, this->NMS_THRESH);
    
    float x1,x2,y1,y2;
    output.object_num = res.size();
    for (size_t j = 0; j < res.size() && j < CV_MAX_OBJECT_NUM; j++) 
    {
        int x_ind = (int)(res[j].bbox[0] / 8);
        int y_ind = (int)(res[j].bbox[1] / 8);
        cv::Rect r = jde_get_rect(input, res[j].bbox);
        x1 = r.x;
        y1 = r.y;
        x2 = r.x + r.width;
        y2 = r.y + r.height;
        x1 = x1 >= input.cols ? input.cols-1 : (x1 <= 0 ? 0 : x1);
        x2 = x2 >= input.cols ? input.cols-1 : (x2 <= 0 ? 0 : x2);
        y1 = y1 >= input.rows ? input.rows-1 : (y1 <= 0 ? 0 : y1);
        y2 = y2 >= input.rows ? input.rows-1 : (y2 <= 0 ? 0 : y2);

        output.object[j].bbox.left_top_x =  x1;
        output.object[j].bbox.left_top_y = y1;
        output.object[j].bbox.w = x2 - x1;
        output.object[j].bbox.h = y2 - y1;
        output.object[j].objectness = res[j].conf;
        output.object[j].classes = (int)res[j].class_id;
        
        feature_map[j].data_host = (void *)(this->host_buffers[2] + y_ind * JDE_OUTPUT2_H * JDE_OUTPUT2_W + x_ind * JDE_OUTPUT2_W);
        feature_map[j].w = JDE_OUTPUT2_W;
        feature_map[j].image_type = CV_FEATUREMAP;
    }
    end = get_current_time();
    printf("nms time cost: %f ms\n", (end - start));
    return 0;
}

