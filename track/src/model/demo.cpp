#include "jde.h"

using namespace nvinfer1;
using namespace cv;
using namespace std;


int main(int argc, char** argv) 
{
    std::string config_Path;
    config_Path = (std::string)("../../../../weight/JDEConfig.yaml");
    cv_object_list output ={0};
    JDETrack track;
    track.Init(config_Path, 1);
    cv::Mat img = cv::imread("../ori_00100.jpg");
    track.Run(img, output);
    for (int i = 0; i < output.object_num; i++)
    {
            cv::Rect rect = cv::Rect((int)output.object[i].bbox.left_top_x,
                (int)output.object[i].bbox.left_top_y,
                (int)output.object[i].bbox.w,
                (int)output.object[i].bbox.h);
            cv::Scalar color = Scalar(0,0,255);
            rectangle(img,rect,color,2);
    }
    cv::imwrite("../res.jpg", img);
    return 0;
}
