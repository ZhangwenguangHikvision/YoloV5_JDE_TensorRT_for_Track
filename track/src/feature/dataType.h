#pragma once
#ifndef DATATYPE_H
#define DATATYPEH

#include <cstddef>
#include <vector>
//#include <Eigen>
#include <Eigen/Core>
#include "sort_debug.h"

#define RUBOST                 0
#define CSTRACK                1

#ifdef SCP_FEATURE
    #define REID_FEATURE_SIZE     8192
#elif RUBOST
    #define REID_FEATURE_SIZE     256
#elif CSTRACK
    #define REID_FEATURE_SIZE     512
#else
    #define REID_FEATURE_SIZE     128
#endif

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, REID_FEATURE_SIZE, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, REID_FEATURE_SIZE, Eigen::RowMajor> FEATURESS;
// typedef Eigen::Matrix<float, 1, 324, Eigen::RowMajor> FEATURE;
// typedef Eigen::Matrix<float, Eigen::Dynamic, 324, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

typedef struct Point{
    int x;
    int y;
}Point;


#endif // DATATYPE_H
