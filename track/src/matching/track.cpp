#include "track.h"

Track::Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int n_init, int max_age, const FEATURE& feature,
    int class_id, float confidence, int max_que_class, int iloop)
{
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = track_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = FEATURESS(1, REID_FEATURE_SIZE);
    // features = FEATURESS(1, 324);
    features.row(0) = feature;//features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = max_age;
    this->outside=true;
    this->counted=false;
    this->class_id = class_id;
    this->confidence = confidence;
    this->iloop = iloop;
    this->class_q.push(class_id);
    this->max_que_class = max_que_class;
}

void Track::predit(KalmanFilter *kf)
{
    /*Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        */

    kf->predict(this->mean, this->covariance);
    this->age += 1;
    this->time_since_update += 1;
}

int Track::get_cross(cv::Point p1,cv::Point p2, cv::Point p){
    return ((p1.x-p.x)*(p2.y-p.y)-(p2.x-p.x)*(p1.y-p.y))/100;
}

void Track::judge_in(cv::Point now_loc,std::deque<cv::Point> area){
    cv::Point pointA = area[0];
    cv::Point pointB = area[1];
    cv::Point pointC = area[2];
    cv::Point pointD = area[3];
    if ((get_cross(pointA, pointB, now_loc)*get_cross(pointC, pointD, now_loc)) < 0
        && (get_cross(pointA, pointC, now_loc)*get_cross(pointB, pointD, now_loc)) < 0){
        this->outside = false;
        this->place_status.push_back(1);
        }
    else if ((get_cross(pointA, pointB, now_loc)) > 0){
        this->outside = true;
        this->place_status.push_back(0);
    }
    else{
        this->outside = true;
        this->place_status.push_back(2);
    }
    if(this->place_status.size()>this->_max_age){
        this->place_status.pop_front();
    }
}

void Track::update_status(std::deque<cv::Point> area){
    cv::Point loc = cv::Point(int(this->mean.data()[0]),int(this->mean.data()[1]));
    judge_in(loc,area);
}

bool Track::get_counted()
{
    return this->counted;
}
void Track::change_counted(bool status)
{   
    this->counted = status;
}

void Track::update(KalmanFilter * const kf, const DETECTION_ROW& detection)
{
    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;

    this->class_id = detection.class_id;
    this->class_q.push(this->class_id);
    if( this->class_q.size() >= this->max_que_class)
    {
        this->class_q.pop();
    }


	this->confidence = detection.confidence;
    this->iloop = detection.iloop;

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if(this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
    cv::Point loc = cv::Point(int(this->mean.data()[0]),int(this->mean.data()[1]));
    this->track_let.push_back(loc);
    if(this->track_let.size()>this->_max_age){
        this->track_let.pop_front();
    }
}

void Track::mark_missed()
{
    if(this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if(this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::is_confirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative()
{
    return this->state == TrackState::Tentative;
}

DETECTBOX Track::to_tlwh()
{
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2)/2);
    return ret;
}

void Track::featuresAppendOne(const FEATURE &f)
{
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size+1, REID_FEATURE_SIZE);
    newfeatures.block(0, 0, size, REID_FEATURE_SIZE) = this->features;
    // FEATURESS newfeatures = FEATURESS(size+1, 324);
    // newfeatures.block(0, 0, size, 324) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}
