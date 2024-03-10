#include "face_tracker.h"

void FaceTracker::trackFaces(cv::Mat& frame, std::vector<DetectedFace>& detectedFaces)
{
    // Initialize the tracker if it's the first frame
    if (trackers.empty()) {
        initializeTrackers(frame, detectedFaces);
    } else {
        // Update the trackers for each detected face
        for (size_t i = 0; i < trackers.size(); ++i) {
            trackers[i]->update(frame, detectedFaces[i].getBoundingBoxUpdate());
        }
    }
}

void FaceTracker::dropTrackers()
{
    trackers.clear();
}

void FaceTracker::initializeTrackers(const cv::Mat& frame, const std::vector<DetectedFace>& detectedFaces)
{
    for (const auto& face : detectedFaces) {
        cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();
        tracker->init(frame, face.getBoundingBox());
        trackers.push_back(tracker);
    }
}