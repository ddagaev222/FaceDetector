#ifndef FACE_TRACKER_H
#define FACE_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <vector>
#include "detected_face.h"

class FaceTracker
{
public:
    FaceTracker() {}

    // Method to track detected faces in each frame
    void trackFaces(cv::Mat& frame, std::vector<DetectedFace>& detectedFaces);

    // Method to clear trackers vector
    void dropTrackers();

private:
    // Method to initialize trackers for detected faces
    void initializeTrackers(const cv::Mat& frame, const std::vector<DetectedFace>& detectedFaces);

private:
    // Private member variable
    std::vector<cv::Ptr<cv::TrackerKCF>> trackers;

};

#endif // FACE_TRACKER_H
