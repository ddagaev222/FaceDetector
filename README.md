# Face Detector C++ app

## Overview

C++ application using OpenCV library for detecting, tracking and recognizing faces in videos.

Implements the following functionality:
- face area detection on video every 10 frames
- tracking of these areas every frame
- creation of face descriptors and saving them to CSV files for each person for further
- creation of feature vector and face recognition

## Usage

Pass the path to train data and test data as console arguments, also you can choose mode
test or live to recognize faces on live cam stream and how to process descriptors (compute or
read from file).

```
face_detector.exe -mode test -desc read -train_path C:\\Projects\\face_detector\\train_data
-test_path C:\\Projects\\face_detector\\test_data

./face_detector -mode live -desc compute -train_path
/home/Projects/face_detector/train_data -test_path /home/Projects/face_detector/test_data
```

At the start of the application there is an opportunity to set your own algorithm parameters, such
as: modelType (kNN or SVM), detectionAlgorithm (haar or LBP), descriptorsAlgorithm (SIFT
or HOG), number of training images and specific parameters of ML algorithms.
