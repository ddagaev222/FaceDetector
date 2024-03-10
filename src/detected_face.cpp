#include "detected_face.h"

cv::Rect& DetectedFace::getBoundingBoxUpdate()
{
    return boundingBox;
}

cv::Rect DetectedFace::getBoundingBox() const
{
    return boundingBox;
}

std::string DetectedFace::getLabel() const
{
    return label;
}

void DetectedFace::setBoundingBox(const cv::Rect& boundingBox)
{
    this->boundingBox = boundingBox;
}

void DetectedFace::setLabel(const std::string& label)
{
    this->label = label;
}
