#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils.h"

class FeatureExtractor
{
public:
    // Method to extract features from a face ROI
    static cv::Mat extractFeatures(const cv::Mat& face, const cv::Mat& kCentres, int dictSize);

    // Method to convert descriptors to features
    static cv::Mat getDataVector(const cv::Mat& descriptors, const cv::Mat& kCenters, int dictSize);

    // Method to save extracted features into a text file
    static void saveDescriptorsToFile(const std::string& filePath, const std::vector<cv::Mat>& descriptors);

    // Method to read feature vectors from a text file
    static std::vector<cv::Mat> readDescriptorsFromFile(const std::string& filePath);

    // Method to read desriptors from Matlab files .mat
    static cv::Mat readDescriptorsFromMatFile(const std::string& filePath, const std::string& descriptorName);

    // Helper method to compute descriptors
    static void computeHOGDescriptors(const cv::Mat& face, std::vector<float>& descriptors);
    static void computeSIFTDescriptors(const cv::Mat& face, cv::Mat& descriptors);

};

#endif // FEATURE_EXTRACTOR_H
