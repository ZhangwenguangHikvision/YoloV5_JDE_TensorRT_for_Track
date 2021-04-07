/******************************************************************
 * created: zhangwenguang
 * date:02/03/2020
 *  version:0.0.1
 *****************************************************************/
#ifndef _OBJECTDETECTTRACK_
#define _OBJECTDETECTTRACK_

#include "CV_AlgCommon.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

/*算法输入，支持批处理, 图像数据由外部管理*/
typedef struct
{
    Mat  image[CV_MAX_BATCH_SIZE];
    unsigned int camera_id[CV_MAX_BATCH_SIZE];    //每张图对应一个id号，相机号从1开始
    int image_num;
} ObjDetectTrackInput;

typedef struct
{
    cv_object_list object_list[CV_MAX_BATCH_SIZE];
    Mat feature_map[CV_MAX_BATCH_SIZE][CV_MAX_OBJECT_NUM];
} ObjDetectTrackOutput;

#ifdef __cplusplus
extern "C" {
#endif

class ObjectDetectTrack
{
    public:
		explicit ObjectDetectTrack();
		~ObjectDetectTrack();
        int Init(const std::string & config_path, const int & max_batch_size);
        int Run(const ObjDetectTrackInput & input, ObjDetectTrackOutput & output);
	private:
        ObjectDetectTrack(const ObjectDetectTrack &);
        const ObjectDetectTrack &operator=(const ObjectDetectTrack &);

        void * contrl;
};


#ifdef __cplusplus
}
#endif

#endif  // _OBJECTDETECTTRACK_
