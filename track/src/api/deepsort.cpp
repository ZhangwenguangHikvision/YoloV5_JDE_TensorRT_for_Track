#include "deepsort.h"
#include "tracker.h"

#define FeatureScale     2
Deep_sort::Deep_sort(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init)
            : m_max_cosine_distance(max_cosine_distance),
            m_nn_budget(nn_budget),
            m_max_iou_distance(max_iou_distance),
            m_max_age(max_age),
            m_n_init(n_init)
{
    // this->track_objects = DS_TrackObjects();
    this->h_tracker = DS_Create(m_max_cosine_distance, m_nn_budget, m_max_iou_distance, m_max_age, m_n_init);

}

Deep_sort::Deep_sort()
{
    // this->track_objects = DS_TrackObjects();
    this->h_tracker = DS_Create(m_max_cosine_distance, m_nn_budget, m_max_iou_distance, m_max_age, m_n_init);

}

DS_Tracker Deep_sort::DS_Create(float max_cosine_distance, 
                        int nn_budget, 
                        float max_iou_distance, 
                        int max_age, 
                        int n_init)
{
    return (DS_Tracker)(new tracker(max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init));
}


bool Deep_sort::DS_Delete(DS_Tracker h_tracker)
{
    delete((tracker *)h_tracker);
    return true;
}

Deep_sort::~Deep_sort()
{
    //printf("Destroy Deep_sort!\n");
    DS_Delete(this->h_tracker);
}

DS_TrackObjects Deep_sort::get_detect_obj()
{
    return this->track_objects;
}

int Deep_sort::get_area_count(){
    return this->area_person;
}
int Deep_sort::get_in_count(){
    return this->in_person;
}
int Deep_sort::get_out_count(){
    return this->out_person;
}

float* Deep_sort::get_hog_feature(cv::Mat img)
{
    cv::Mat resize_img;
    cv::HOGDescriptor hog = cv::HOGDescriptor(cvSize(20, 20), cvSize(10, 10), cvSize(5, 5), cvSize(5, 5), 9);
    cv::resize(img, resize_img, cv::Size(30,30),(0, 0), (0, 0), cv::INTER_LINEAR);
    std::vector<float> descriptors;
    // float *descriptors;
    hog.compute(resize_img, descriptors, cv::Size(20, 20), cv::Size(0, 0));
    float *feature_float = (float *)malloc(descriptors.size()*sizeof(float));
    assert(feature_float);
    for(int i=0;i<REID_FEATURE_SIZE;i++)
    {
        feature_float[i]=descriptors[i*2];
    }
    // FEATURE *feature = new FEATURE(feature_float);

    // delete feature_float;
    // feature_float = nullptr;

    return feature_float;
}

bool Deep_sort::update(
    // DS_Tracker h_tracker, 
    DS_DetectObjects detect_objects, 
    // DS_TrackObjects &track_objects,
    std::deque<cv::Point> area,
    cv::Mat img, int max_que_class)
{
    if(NULL == this->h_tracker)
    {
        printf("NULL pt !\n");
        return false;
    }
    tracker *p_tracker=(tracker *)this->h_tracker;
    DETECTION_ROW temp_object;
    DETECTIONS detections;
    
    for(int iloop=0;iloop<detect_objects.size();iloop++)
    {
        temp_object.class_id=detect_objects[iloop].class_id;
        temp_object.confidence=detect_objects[iloop].confidence;
        temp_object.tlwh = DETECTBOX(
            detect_objects[iloop].rect.x, 
            detect_objects[iloop].rect.y, 
            detect_objects[iloop].rect.width, 
            detect_objects[iloop].rect.height);
            // temp_object.feature.setZero();
            if((detect_objects[iloop].rect.x < 0 || detect_objects[iloop].rect.y < 0)||
               ( (detect_objects[iloop].rect.x + detect_objects[iloop].rect.width) > img.size[1])||
               ((detect_objects[iloop].rect.y + detect_objects[iloop].rect.height) > img.size[0]))
            {
                printf("x = %d, y = %d,w = %d,h = %d, image_w = %d, image_h = %d\n", detect_objects[iloop].rect.x,detect_objects[iloop].rect.y,
                detect_objects[iloop].rect.width,detect_objects[iloop].rect.height, img.size[1], img.size[0]);
                return false;
            }
            try{
                float *feature = get_hog_feature(img(cv::Rect(detect_objects[iloop].rect.x,detect_objects[iloop].rect.y,detect_objects[iloop].rect.width,detect_objects[iloop].rect.height)));
                temp_object.feature = FEATURE(feature);
                if(feature)
                {
                    delete feature;
                    feature = nullptr;
                }    
            }
            catch(std::exception &e){
                printf("ERROR: bboxes = %d %d %d %d\n", detect_objects[iloop].rect.x, detect_objects[iloop].rect.y, detect_objects[iloop].rect.width, detect_objects[iloop].rect.height);
                temp_object.feature.setZero();
            }
            detections.push_back(temp_object);

    }
    p_tracker->predict();
    p_tracker->update(detections, max_que_class);
    DETECTBOX output_box;
    DS_TrackObject track_object;
    this->track_objects.clear();
    this->area_person = 0;
    for(Track& track : p_tracker->tracks) 
    {
        if(!track.is_confirmed() || track.time_since_update > 1) 
            continue;
        output_box=track.to_tlwh();
        
        track_object.track_id=track.track_id;
        // track_object.class_id=track.class_id;
        // track_object.confidence=track.confidence;
        track_object.rect.x=output_box(0);
        track_object.rect.y=output_box(1);
        track_object.rect.width=output_box(2);
        track_object.rect.height=output_box(3);
        track_object.outside=track.outside;
        
        // track.track_let.push_back(cv::Point(int(output_box(0)+output_box(2)/2.0),int(output_box(1)+output_box(3)/2.0)));
        track_object.tracklet=track.track_let;

        if(area.size()==4){
            track.update_status(area);
            if(track.place_status.back()==2){
                std::deque<int>::iterator pos=find(track.place_status.begin(),track.place_status.end(),1);
                if(pos!=track.place_status.end() && track.get_counted()==false)
                {
                    this->out_person++;
                    // track.counted==true;
                    track.change_counted(true);
                }
            }
            if(track.place_status.back()==1){
                this->area_person++;
            }
        }
        this->track_objects.push_back(track_object);
    }
    return true;
}

bool Deep_sort::update(DS_DetectObjects detect_objects,  cv::Mat img, cv_image * feature_map_ptr, int max_que_class)
{
    if(NULL == this->h_tracker)
    {
        printf("NULL pt !\n");
        return false;
    }
 
    tracker *p_tracker=(tracker *)this->h_tracker;
    DETECTION_ROW temp_object;
    DETECTIONS detections;
#ifdef MY_DEBUG
    clock_t start,end;
    start = clock();
#endif
    for(int iloop=0;iloop<detect_objects.size();iloop++)
    {
        //这里直接赋值属性，因为跟踪仅仅针对同一类别跟踪
        temp_object.class_id=detect_objects[iloop].attribute;
        temp_object.confidence=detect_objects[iloop].confidence;
        temp_object.iloop = iloop;      //标记在原目标链表的顺序id
        temp_object.tlwh = DETECTBOX(
            detect_objects[iloop].rect.x, 
            detect_objects[iloop].rect.y, 
            detect_objects[iloop].rect.width, 
            detect_objects[iloop].rect.height);
            // temp_object.feature.setZero();
            if((detect_objects[iloop].rect.x < 0 || detect_objects[iloop].rect.y < 0)||
               ( (detect_objects[iloop].rect.x + detect_objects[iloop].rect.width) > img.size[1])||
               ((detect_objects[iloop].rect.y + detect_objects[iloop].rect.height) > img.size[0]))
            {
                printf("x = %d, y = %d,w = %d,h = %d, image_w = %d, image_h = %d\n", detect_objects[iloop].rect.x,detect_objects[iloop].rect.y,
                detect_objects[iloop].rect.width,detect_objects[iloop].rect.height, img.size[1], img.size[0]);
                return false;
            }
        
            if((feature_map_ptr[iloop].image_type == CV_FEATUREMAP) 
                && (feature_map_ptr[iloop].w > 0))
            {
                if(feature_map_ptr[iloop].w != REID_FEATURE_SIZE)
                {
                    printf("deepsort error: feature_map_ptr[iloop].w != REID_FEATURE_SIZE: %d != %d! \n", 
                        feature_map_ptr[iloop].w, REID_FEATURE_SIZE);
                    return false;
                }
                float * feature_float = (float *)feature_map_ptr[iloop].data_host;
                temp_object.feature = FEATURE(feature_float);
            }
            else
            {
                 try
                 {
                    float *feature = get_hog_feature(img(cv::Rect(detect_objects[iloop].rect.x,detect_objects[iloop].rect.y,detect_objects[iloop].rect.width,detect_objects[iloop].rect.height)));
                    temp_object.feature = FEATURE(feature);
                    if(feature)
                    {
                        delete feature;
                        feature = nullptr;
                    }    
                }
                catch(std::exception &e)
                {
                    printf("bboxes = %d %d %d %d\n", detect_objects[iloop].rect.x, detect_objects[iloop].rect.y, detect_objects[iloop].rect.width, detect_objects[iloop].rect.height);
                    temp_object.feature.setZero();
                }
            }
            detections.push_back(temp_object);

    }
#ifdef MY_DEBUG
    end = clock();
    printf("feature time cost: %f ms!\n", (double)(end-start)/CLOCKS_PER_SEC *1000);
    start = clock();
#endif
    p_tracker->predict();
    p_tracker->update(detections, max_que_class);
    DETECTBOX output_box;
    DS_TrackObject track_object;
    this->track_objects.clear();
    this->area_person = 0;
    for(Track& track : p_tracker->tracks) 
    {
        if(!track.is_confirmed() || track.time_since_update > 1) 
            continue;
        output_box=track.to_tlwh();
        
        track_object.track_id=track.track_id;
        int cls_id = 0;
        //遍历
        int size = track.class_q.size();
        //printf("-----------------size %d------track.track_id%d  -------------------\n", size, track.track_id);
        for(int i = 0; i < size; i++)
        {
            int front = track.class_q.front();
            cls_id += front - (int)PERSON_MALE;          //这样写仅仅支持2分类属性
            track.class_q.pop();
            track.class_q.push(front);
            //printf("front %d\n", front);
        }
        if(cls_id>(size/2))
        {
            track_object.class_id = PERSON_FEMALE;
        }
        else
        {
            track_object.class_id = PERSON_MALE;
        }
        //printf("---------------track_object.class_id %d  track.class_id %d--------\n", track_object.class_id, track.class_id);
        track_object.class_id=track.class_id;
        track_object.confidence=track.confidence;
        track_object.iloop = track.iloop;
        track_object.rect.x=output_box(0);
        track_object.rect.y=output_box(1);
        track_object.rect.width=output_box(2);
        track_object.rect.height=output_box(3);
        track_object.outside=track.outside;
        
        // track.track_let.push_back(cv::Point(int(output_box(0)+output_box(2)/2.0),int(output_box(1)+output_box(3)/2.0)));
        track_object.tracklet=track.track_let;
        track_object.current_feature = track.current_feature;
        this->track_objects.push_back(track_object);
    }
    return true;
}
