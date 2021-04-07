/******************************************************************
 * 定义通用结构体
 * created: zhangwenguang
 * date:11/04/2019
 * version:0.0.1
 * 
 * date:04/26/2020
 * version:0.0.2 add human keypoint by zhangwenguang
 * 
 * date:03/07/2020
 * version:0.0.3 modifyed by zhangwenguang
 *****************************************************************/
#ifndef _CVALGCOMMON_
#define _CVALGCOMMON_

#include "CV_AlgType.h"
#include "CV_ErrorCode.h"

#ifdef __cplusplus
extern "C" {
#endif
/*用于添加计算机视觉通用结构体*/

/*图像结构体*/
/*如果用于表示featuremap，则w表示维度，h，c均为1*/
typedef struct cv_image 
{
    int w;                                                      //宽度
    int h;                                                      //高度
    int c;                                                      //通道数
    unsigned int camera_id;               //图像来自哪个相机
    ImageType image_type;              //图像类型
    void *data_host;                             //cpu端内存地址
	void *data_dev;                              //设备端内存地址
} cv_image;                                           //size: 28

/*目标框结构体*/
typedef struct cv_box 
{
    float left_top_x;                              //左上角x坐标
    float left_top_y;                             //左上角y坐标
    float w;                                              //宽度
    float h;                                               //高度
    float angle;                                      //目标框角度 极坐标 用于支持斜框
} cv_box;                                              //size: 16

/*目标关键点信息*/
typedef struct cv_keypoint 
{
    float x;                              //点x坐标
    float y;                             //点y坐标
    float score;                   //置信度
} cv_keypoint;                 //size: 12

/*多边形结构体, 点从0开始，顺时针排列*/
typedef struct cv_polygon
{
    int point_num;
    cv_keypoint polygon[CV_MAX_KEYPOINT_NUM];
}cv_polygon;                         //size: 304
               
/*目标结构体*/
typedef struct cv_object
{
    cv_box bbox;                               //目标框
    int classes;                                   //类别
    float objectness;                       //置信度       
    unsigned int track_id;            //有效值从1开始   跟踪ID
    unsigned int camera_id;       //有效值从1开始   跨镜头ID
    float *prob;                                 //各类型置信度
    AttributeType attribute;        //目标属性属性
    cv_keypoint kp_info[CV_MAX_KEYPOINT_NUM];    //关键点信息
    int reserved[7];
} cv_object;       

/*目标链表结构体*/
typedef struct cv_object_list
{
    cv_object object[CV_MAX_OBJECT_NUM];
    int object_num;                               //有效目标个数
    int reserved[63];
}cv_object_list;    

/*模型信息结构体*/
typedef struct cv_model
{
    char cfg_file_path[1024];                       //配置文件路径
	char weight_file_path[1024];               //权重文件路径
	char* model_buff;                                     //模型内存地址
    int reserved;
}cv_model;                                                      

#ifdef __cplusplus
}
#endif

#endif  // _CVALGCOMMON_
