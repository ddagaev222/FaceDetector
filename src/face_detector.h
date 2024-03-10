#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <stdexcept>

#include "utils.h"
#include "detected_face.h"
#include "face_tracker.h"

class FaceDetector
{
public:
    FaceDetector(const std::string& cascadeFilePath, const std::string& selectedAlgorithm = "haar")
                : cascadeFilePath{cascadeFilePath}, selectedAlgorithm{selectedAlgorithm} {}

    // Method to detect faces in a single image
    std::vector<DetectedFace> detectFaces(const cv::Mat& image, const std::string& label = "Unknown");

    // Setter method to select the face detection algorithm
    void selectAlgorithm(const std::string& algorithm);

private:
    // Method to detect faces using Haar cascade classifier
    void detectFacesWithHaarCascade(const cv::Mat& image, std::vector<DetectedFace>& detectedFaces, const std::string& label);

    // Method to detect faces using LBP cascade classifier
    void detectFacesWithLBP(const cv::Mat& image, std::vector<DetectedFace>& detectedFaces);

private:
    // Member variable to store the selected face detection algorithm
    std::string selectedAlgorithm;
    std::string cascadeFilePath;

};

#endif // FACE_DETECTOR_H