/******************************************************************
 * Copyright (C) 2020 by ZheJiang Lab. All rights reserved.
 * 示例代码
 * created: zhangwenguang
 * date:06/30/2020
 *  version:0.0.1
 *****************************************************************/

#include  "CV_ObjectDetectTrack.h"
#include "time.h"
#include <sys/time.h>
#include <iostream>

// ====================================================================
static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000.0 + tv.tv_usec/1000.0;
}


int main(int argc, char *argv[])
{
    std::string config_Path;
    double start,end;
    int ret;

    config_Path = (std::string)("../../weight/JDEConfig.yaml");
    ObjectDetectTrack trackor;
    ret = trackor.Init(config_Path, 1);
    if(CODE_CV_OK != ret)
    {
        printf("ERROR!\n");
        return -1;
    }
    else
    {
        printf("trackor init sucessed!\n");
    }

    // inference
    VideoCapture capture;
    capture.open("../../../DeepSort_yoloV3-HOG_feature/data/a3.mp4");
    if(!capture.isOpened())
    {
        printf("can not open ...\n");
        return -1;
    }

    Size size = Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer;
    writer.open("../../result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, size, true);
    Mat frame;
    int frame_cnt = 0;
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    char text[125];

    while (capture.read(frame)) 
    {
        if(frame.cols <= 0 || frame.data == NULL)
        {
            break;
        }
        //cv::flip(frame,frame,1);
        frame_cnt = frame_cnt + 1;
        if (frame_cnt % 1 == 0) 
        {
            
        }
        else 
        { 
            continue; 
        }

        ObjDetectTrackOutput track_output;
        ObjDetectTrackInput track_input;
        track_input.image_num = 1;
        track_input.image[0] = frame;
        //track
        start = get_current_time();
        ret = trackor.Run(track_input, track_output);
        end = get_current_time();
        if(CODE_CV_OK != ret)
        {
            printf("Run ERROR!\n");
            return -1;
        }
        else
        {
            printf("trackor.Run time cost: %f ms!\n", (double)(end-start));
        }

        for (int i = 0; i < track_output.object_list[0].object_num; i++)
        {
                Rect rect = Rect((int)track_output.object_list[0].object[i].bbox.left_top_x,
                    (int)track_output.object_list[0].object[i].bbox.left_top_y,
                    (int)track_output.object_list[0].object[i].bbox.w,
                    (int)track_output.object_list[0].object[i].bbox.h);
                cv::Scalar color = Scalar(0,0,255);
                rectangle(frame,rect,color,2);
                memset(text, 0, sizeof(text));
                sprintf(text,"%d",track_output.object_list[0].object[i].track_id);
                Point point  = Point((int)track_output.object_list[0].object[i].bbox.left_top_x,
                    (int)track_output.object_list[0].object[i].bbox.left_top_y);
                cv::putText(frame, text, point, font_face, 0.7, color, 1, 4, 0);
                //std::cout<< track_output.feature_map[0][i]<<std::endl;
        }
        writer.write(frame);
    }
    return 0;
}
