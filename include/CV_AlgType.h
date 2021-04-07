/******************************************************************
 * 定义通用宏，类型
 * created: zhangwenguang
 * date:11/08/2019
 * version:0.0.1
 *****************************************************************/
#ifndef _CVALGTYPE_
#define _CVALGTYPE_


/*普通场景目标链表包含的最大目标个数*/
#define CV_MAX_OBJECT_NUM  128
/*拥挤场景目标链表包含的最大目标个数*/
#define CV_MAX_OBJECT_NUM_CROWD  512
/*批处理最大个数*/
#define CV_MAX_BATCH_SIZE  32
/*人体关键点最大个数*/
#define CV_MAX_KEYPOINT_NUM  25


#define  ZJ_MAX(x,y)    ((x) > (y) ? (x) : (y))
#define  ZJ_MIN(x,y)    ((x) < (y) ? (x) : (y))
#define  ZJ_CLIP(x, low, high) ( x > high ? high : (x < low ? low : x))

/*算法类型枚举*/
/*****************************************************
 * 不同的枚举类型表示的图像像素排列方式不一样
 * CV_RGB: RGB, RGB, RGB......
 * CV_BGR: BGR, BGR, BGR......(opencv打开的默认方式)
 * CV_RGBP: RRR......, GGG......, BBB......
 * CV_BGRP: BBB......, GGG......, RRR......
 * CV_YUV444: YYY......, UUU......, VVV......
 * CV_YUV422P: YYYYYYYY VVVV UUUU
 * CV_I420: YYYYYYYY UU VV
 * CV_NV21: YYYYYYYY UVUV
 * CV_NV12: YYYYYYYY VUVU
 * CV_FEATUAEMAP: 表示输入的格式为featuremap
*****************************************************/
typedef enum ImageType
{
    CV_RGB = 0,
    CV_BGR,
    CV_RGBP,
    CV_BGRP,
    
    CV_YUV444 = 50,
    CV_YUV422P,
    CV_I420,
    CV_NV21,
    CV_NV12,

    CV_FEATUREMAP=100,

    CV_OPENCVMAT = 120             //使用opencv的mat格式
} ImageType;

/*属性类型枚举*/
typedef enum AttributeType
{
    /*人属性*/
    PERSON_HAT = 0,                          //戴帽子
    PERSON_KNAPSACK,                  //背包
    PERSON_PRONT,                         //正面朝向镜头
    PERSON_BACK,                            //背面朝向镜头
    PERSON_MALE,                            //男性
    PERSON_FEMALE,                       //女性

    /*车属性*/
    VEHICLE_WHITE = 100,               //白色车

} AttributeType;


#endif  // _CVALGTYPE_
