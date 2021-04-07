#ifndef _DEEP_SORT_H_
#define _DEEP_SORT_H_

#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include "CV_AlgCommon.h"
#include "dataType.h"
//#ifdef __cplusplus
//extern "C" {
//#endif

typedef struct
{
	int x;
	int y;
	int width;
	int height;
}DS_Rect;

typedef struct
{
	int class_id;
	AttributeType attribute;
	DS_Rect rect;
	float confidence;
}DS_DetectObject;

typedef struct
{
	int track_id;
	int class_id;
	int iloop;
	float confidence;
	DS_Rect rect;
	std::deque<cv::Point> tracklet;
	bool outside;
	FEATURE current_feature;
	AttributeType attribute;
}DS_TrackObject;


typedef void * DS_Tracker;
typedef std::vector<DS_DetectObject> DS_DetectObjects;
typedef std::vector<DS_TrackObject> DS_TrackObjects;


class Deep_sort{
    public:
		Deep_sort(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init);
		Deep_sort();
		~Deep_sort();
		int get_area_count();
		int get_in_count();
		int get_out_count();
		bool update(DS_DetectObjects detect_objects, std::deque<cv::Point> area,cv::Mat img, int max_que_class);
		bool update(DS_DetectObjects detect_objects,  cv::Mat img, cv_image * feature_map_ptr, int max_que_class);
		DS_TrackObjects get_detect_obj();
	private:
		int area_person=0;
		int in_person=0;
		int out_person=0;
		float m_max_cosine_distance=0.2;
		int m_nn_budget=100;
		float m_max_iou_distance=0.7; 
		int m_max_age=30;
		int m_n_init=3;
		DS_Tracker h_tracker;
		DS_TrackObjects track_objects;
		DS_Tracker DS_Create(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init);
		bool DS_Delete(DS_Tracker h_tracker);
		float* get_hog_feature(cv::Mat img);
};

//#ifdef __cplusplus
//}
//#endif

#endif


