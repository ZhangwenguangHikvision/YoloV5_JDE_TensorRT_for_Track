#include "CV_ObjectDetectTrack.h"
#include "deepsort.h"
#include "jde.h"

/*跟踪算法结构体*/
typedef struct
{
    unsigned int fps;
    JDETrack *DetectPtr;
    Deep_sort *TrackePtr;
    int max_que_class;
    int fps_control;
} ObjTrackHdl; 

static DS_DetectObjects det2detobj(const cv_object_list *dets,int width,int height)
{
    DS_DetectObjects result;
    result.clear();
    for (int i=0;i<dets->object_num;i++)
    {
        DS_Rect rec;
        DS_DetectObject obj;
        int x = (int)(dets->object[i].bbox.left_top_x);
        int y = (int)(dets->object[i].bbox.left_top_y);
        int w = (int)(dets->object[i].bbox.w);
        int h = (int)(dets->object[i].bbox.h);

        if(x<0) 
            x=0;
        if(y<0) 
            y=0;
        if((x+w)>width) 
            w=width-x;
        if((y+h)>height) 
            h=height-y;
        rec.x = x;
        rec.y = y;
        rec.width = w;
        rec.height = h;
        obj.class_id = dets->object[i].classes;
        obj.rect = rec;
        obj.confidence = dets->object[i].objectness;
        obj.attribute= dets->object[i].attribute;
        result.push_back(obj);
    }
    return result;
}

ObjectDetectTrack::ObjectDetectTrack()
{
    this->contrl = NULL;
    return;
}

int ObjectDetectTrack::Init(const std::string & config_path, const int & max_batch_size)
{
    ObjTrackHdl * objtrack_ptr = new ObjTrackHdl;
    objtrack_ptr->DetectPtr = new JDETrack();
    objtrack_ptr->DetectPtr->Init(config_path, max_batch_size);
    objtrack_ptr->TrackePtr = new Deep_sort();
    objtrack_ptr->fps_control = 12;
    objtrack_ptr->max_que_class = 9;
    objtrack_ptr->fps = 25;
    this->contrl = (void *)objtrack_ptr;
    return CODE_CV_OK;
}

int ObjectDetectTrack::Run(const ObjDetectTrackInput & input, ObjDetectTrackOutput & output)
{
    CODECVASSERT( NULL != this->contrl, CODE_CV_NULL_POINTER);
    ObjTrackHdl * objtrack_ptr = (ObjTrackHdl *)this->contrl;
    CODECVASSERT( NULL != objtrack_ptr->DetectPtr, CODE_CV_NULL_POINTER);
    CODECVASSERT( NULL != objtrack_ptr->TrackePtr, CODE_CV_NULL_POINTER);
    JDETrack * jde_ptr = objtrack_ptr->DetectPtr;
    Deep_sort * deepsort_ptr = objtrack_ptr->TrackePtr;
    for(int i = 0; i < input.image_num; i++)
    {
        DS_DetectObjects detect_objects;
        DS_TrackObjects track_objects;
        cv_object_list detect_output ={0};
        cv_image feature_map[CV_MAX_OBJECT_NUM];
        cv::Mat frame = input.image[i];

        //检测与提取特征
        jde_ptr->Run(frame, detect_output, feature_map);

        int imag_w =  input.image[i].cols;
        int imag_h =  input.image[i].rows;
        track_objects.clear();
        detect_objects = det2detobj(&detect_output, imag_w, imag_h);
        deepsort_ptr->update(detect_objects, frame, feature_map, objtrack_ptr->max_que_class);
        track_objects = deepsort_ptr->get_detect_obj();
        output.object_list[i].object_num = track_objects.size();

        for(int j = 0; j < track_objects.size(); j++)
        {
            output.object_list[i].object[j].track_id = track_objects[j].track_id;
            output.object_list[i].object[j].bbox.left_top_x = track_objects[j].rect.x;
            output.object_list[i].object[j].bbox.left_top_y = track_objects[j].rect.y;
            output.object_list[i].object[j].bbox.w = track_objects[j].rect.width;
            output.object_list[i].object[j].bbox.h = track_objects[j].rect.height;
            output.object_list[i].object[j].attribute = (AttributeType)track_objects[j].class_id;
            output.object_list[i].object[j].objectness = track_objects[j].confidence;
        }
    }
    return CODE_CV_OK;
}

ObjectDetectTrack::~ObjectDetectTrack()
{
    if( NULL == this->contrl)
    {
        printf("ERROR:this->contrl is null!\n");
        return;
    }
    ObjTrackHdl * alg_hdl = (ObjTrackHdl *)this->contrl;
    if(alg_hdl->DetectPtr)
    {
        delete(alg_hdl->DetectPtr);
        alg_hdl->DetectPtr = NULL;
    }
    if(alg_hdl->TrackePtr)
    {
        delete(alg_hdl->TrackePtr);
        alg_hdl->TrackePtr = NULL;
    }
    delete(alg_hdl);
    this->contrl = NULL;
    return;
}
