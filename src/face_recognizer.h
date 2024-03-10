#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>

#include "detected_face.h"
#include "face_detector.h"
#include "feature_extractor.h"
#include "utils.h"
#include "annotation.h"

struct QualityMetrics
{
    int elementsCount = 0;
    int mTP = 0;
    int mFP = 0;
    int mFN = 0;
    double mDetectionError = 0.0;
    double mAccuracy = 0.0;
};

class FaceRecognizer {
public:
    // Constructor:  modelType is ["kNN" or "SVM"] used for solving classification problem
    // and selectedAlgorithm is ["HaarCascade" or "LBP"] used in Face Detector to compute descriptors
    FaceRecognizer(const std::string& trainDataPath, const std::string& cascadeFilePath,
                   const std::string& modelType = "knn", const std::string& detectionAlgorithm = "haar",
                   const std::string& descriptorsAlgorithm = "sift", int dictSize = 230);

    // Method to train the face recognizer with training data
    void train(int attempts, int iterationNumber, int trainImagesNumber, bool compute = true,  bool show = false,
               bool save = false, const std::string& descFilename = "descriptors.csv", bool quality = false, const std::string& annotFilename = "annotation.csv");

    // Method to recognize subject on the test dataset
    void runTest(const std::string& testPath, bool show = true, int k = 5, float thresholdDistance = 50, const std::string& annotationFilename = "annotation.csv");

    // Method to recognize subject on the VideoCapture from WebCam LiveStream
    void runLive(int k = 5, float thresholdDistance = 50);

private:
    // Method to compute descriptors
    void computeTrainDescriptors(const std::string& className, int imagesNumber, int classLabel, cv::Mat& allDescriptors,
                                 std::vector<cv::Mat>& allDescPerImg, std::vector<int>& allClassPerImg, int& allDescPerImgNum, bool showResults,
                                 bool saveToFile, const std::string& descFilename, bool quality, const std::string& annotFilename);

    // Method to prepare descriptors for all subjects in train directory and save them in CSV files
    void computeAndPrepareDescriptors(int imagesNumber, cv::Mat& allDescriptors, std::vector<std::string>& allClassNames,
                                      std::vector<cv::Mat>& allDescPerImg, std::vector<int>& allClassPerImg, int& allDescPerImgNum, bool showResults,
                                      bool saveToFile, const std::string& descFilename, bool quality, const std::string& annotFilename);

    // Method to read descriptors from existing CSV files
    void readDescriptors(const std::string& filename, cv::Mat& allDescriptors, std::vector<std::string>& allClassNames,
                         std::vector<cv::Mat>& allDescPerImg, std::vector<int>& allClassPerImg, int& allDescPerImgNum);

    // Method to compute distance between two face's rects
    double computeDistance(const cv::Rect& rect1, const cv::Rect& rect2);

    // Method to compute text position relative to Face Rect
    void computeTextPosition(cv::Point& textPoint, const cv::Size& textSize, int rows, int cols);

    // Method to detect Faces on the video and predict thier classes
    void testPerClass(const std::string& classPath, bool show, int k, float thresholdDistance, const std::string& annotationFilename);

    // Method to compare detected Faces with its text annotations
    void computeMetrics(const std::vector<std::vector<DetectedFace>> foundFaces, const std::vector<DetectedFace> trueFaces,
                        QualityMetrics& metrics, double eps = 30.0);

    // Method to compute Quality Metrics on the whole dataset
    void computeMeanMetricsAndPrint(const std::vector<QualityMetrics>& metrics) const;

    // Print Quality Metrics to console
    void printMetrics(const QualityMetrics& metrics) const;

private:
    std::string modelType;

    cv::Ptr<cv::ml::StatModel> pModel;
    cv::Ptr<FaceDetector> pFaceDetector;
    cv::Ptr<FaceTracker> pTracker;

    std::string trainDataPath;

    std::vector<QualityMetrics> trainMetrics;
    std::vector<QualityMetrics> testMetrics;

    std::vector<std::string> allClassNames;
    std::vector<int> allClassLabels;

    bool trained;

    // Initial clusters centers
    cv::Mat kCenters;

    // Model properties
    int dictSize;
    std::string detectionAlgorithm;
    std::string descriptorsAlgorithm;

    // Text properties
    int fontText = cv::FONT_HERSHEY_SIMPLEX;
    double textScale = 0.4;
    double thickness = 1.3;
    cv::Scalar textColor = cv::Scalar(0, 255, 0);
    int baseLine = 0;
};

#endif // FACE_RECOGNIZER_H