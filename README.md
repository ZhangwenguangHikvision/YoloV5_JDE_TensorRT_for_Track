# YoloV5_JDE_TensorRT_for_Track

## Introduction
A multi object detect and track Library Based on tensorrt

一个基于TensorRT的多目标检测和跟踪融合算法库，可以同时支持行人的多目标检测和跟踪，当然也可以仅仅当检测库使用。

本算法的主框架采用了JDE+deepsort结构，其中由JDE算法检测出人的坐标与其对应的外观特征，然后基于deepsort的方法进行目标与运动轨迹的匹配。
JDE中的检测框架则采用了YOLOV5 L 的模型结构。

CSTrack3_0.yaml为本网络的模型结构，模型训练的代码大部分借鉴了CSTrack原文作者的开源项目，这里不再开源，大家有兴趣可以阅读CSTrack论文。
需要注意的是，本项目由于追求速度将CStrack的CCN和SAAN模块改成了JDE模块，也就是直接在anchor上提取reid特征并没有进行detect和reid的解耦模块。
如果读者需要的话可以自行修改，这样可以提升IDswich方面的性能。

## Reference
* JDE: https://arxiv.org/pdf/1909.12605v1.pdf
* CSTrack: https://arxiv.org/pdf/2010.12138.pdf
* YOLOV5: https://github.com/ultralytics/yolov5
* DeepSort: https://arxiv.org/pdf/1703.07402.pdf

## Requirements
* ubuntu 18.04 for x86
* gcc/g++ >= 7.5.0  
* opencv >= 3.4.8
* cuda >=10.0  cudnn >= 7.6
* tensorRT >= 7.0.0
* (Optional) ffmpeg (used in the video demo)

## How to build and run
* modify track/CMakeLists.txt Change opencv and tensorRT to your local directory
* modify demo/CMakeLists.txt Change opencv and tensorRT to your local directory
* modify demo/src/main.cpp Change video path to your local directory
* sh make.sh
* cd demo/build
* ./itest

## How to convert to tensort gie file
* cd PytorchToTensorRTGIE
* modify CMakeLists.txt Change opencv and tensorRT to your local directory
* download jde.wts file
* cd build
* cmake ..
* make
* ./yolov5 -s
* #Verify detect results
* ./yolov5 -d ../sample/              

## Model
* TensorRT GIE  Model File:
[[Baidu]](https://pan.baidu.com/s/1iYL3iV_qzJaE3GXn1S4NNg)  key: 6yc6. 

Download the model and put it to /weight/

* .wts File:
[[Baidu]](https://pan.baidu.com/s/1KCp8og13vPYad9OqOCvN3w)  key: c30n. 

Download the model and put it to /PytorchToTensorRTGIE/

## Video Demo
<img src="assets/demo.gif" width="800"/>

## Acknowledgement
A large portion of code is borrowed from [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [sephirothhua/DeepSort_yoloV3-HOG_feature](https://github.com/sephirothhua/DeepSort_yoloV3-HOG_feature) and , many thanks to their wonderful work!
