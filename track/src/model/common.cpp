#include "common.hpp"

using namespace nvinfer1;

cv::Mat jde_preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = Yolo_Jde::INPUT_W / (img.cols*1.0);
    float r_h = Yolo_Jde::INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = Yolo_Jde::INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo_Jde::INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = Yolo_Jde::INPUT_H;
        x = (Yolo_Jde::INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo_Jde::INPUT_H, Yolo_Jde::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::cuda::GpuMat jde_cuda_resize_img(const cv::cuda::GpuMat& img, int input_w, int input_h) 
{
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h* img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::cuda::GpuMat re(h, w, CV_8UC3);
    cv::cuda::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::cuda::GpuMat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


int jde_cuda_preprocess(cv::Mat& img, void* indata, int input_w, int input_h)
{
    void* data_tmp = indata;
    cv::cuda::GpuMat orig_img = cv::cuda::GpuMat(img);
    cv::cuda::GpuMat resized_img = jde_cuda_resize_img(orig_img, input_w, input_h);
    cv::cuda::cvtColor(resized_img, resized_img, CV_BGR2RGB);
    cv::cuda::GpuMat imagePreproc;
    resized_img.convertTo(imagePreproc, CV_32FC3, 1 / 255.0);

    cv::cuda::GpuMat cuda_bgr[3];
    cv::cuda::split(imagePreproc, cuda_bgr);//split source
    cv::Mat bgr[3];
    void *ptr1;
    for (int i = 0; i < 3; i++) 
    {
        cuda_bgr[i].download(bgr[i]);
        ptr1 = (bgr[i].data);
        memcpy((void *) data_tmp, (void *) ptr1, input_h * input_w * sizeof(float));
        data_tmp = (float *) data_tmp + input_h * input_w;
    }
    return 0;
}


cv::Rect jde_get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo_Jde::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo_Jde::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo_Jde::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo_Jde::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo_Jde::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo_Jde::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Yolo_Jde::Detection& a, const Yolo_Jde::Detection& b) {
    return a.conf > b.conf;
}

void jde_nms(std::vector<Yolo_Jde::Detection>& res, float *output, float conf_thresh, float nms_thresh) 
{
    int det_size = sizeof(Yolo_Jde::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo_Jde::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo_Jde::MAX_OUTPUT_BBOX_COUNT; i++) 
    {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo_Jde::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo_Jde::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) 
    {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) 
        {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) 
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) 
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void jde_cuda_doInference(IExecutionContext& context, cudaStream_t& stream, void **device_buffers, 
    float** host_buffers, int batchSize, const int INPUT_H, const int INPUT_W, const int OUTPUT_SIZE,
    const int OUTPUT_SIZE2) 
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(device_buffers[0], host_buffers[0], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, device_buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(host_buffers[1], device_buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(host_buffers[2], device_buffers[2], batchSize * OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}