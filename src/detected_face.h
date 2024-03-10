#ifndef DETECTED_FACE_H
#define DETECTED_FACE_H

#include <opencv2/opencv.hpp>
#include <string>

class DetectedFace
{
public:
    // Constructors
    DetectedFace() = default;
    DetectedFace(const cv::Rect& boundingBox, const std::string& label) : boundingBox(boundingBox), label(label) {}

    // Getter methods
    cv::Rect& getBoundingBoxUpdate();
    cv::Rect getBoundingBox() const;
    std::string getLabel() const;

    // Setter methods
    void setBoundingBox(const cv::Rect& boundingBox);
    void setLabel(const std::string& label);

private:
    // Private member variables
    cv::Rect boundingBox; // Bounding box of the detected face
    std::string label;    // Label (person's name) associated with the face
};

#endif // DETECTED_FACE_H
