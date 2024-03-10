#include "face_detector.h"

using namespace std;

vector<DetectedFace> FaceDetector::detectFaces(const cv::Mat& image, const std::string& label)
{
    vector<DetectedFace> detectedFaces;

    // Check if the input image is empty
    if (image.empty()) {
        throw invalid_argument("Input image is empty");
    }

    // Perform face detection using the selected algorithm
    if (selectedAlgorithm == "haar") {
        detectFacesWithHaarCascade(image, detectedFaces, label);
    } else if (selectedAlgorithm == "lbp") {
        detectFacesWithLBP(image, detectedFaces);
    } else {
        throw invalid_argument("Invalid face detection algorithm");
    }

    return detectedFaces;
}

void FaceDetector::selectAlgorithm(const string& algorithm)
{
    selectedAlgorithm = algorithm;
}

void FaceDetector::detectFacesWithHaarCascade(const cv::Mat& frame, vector<DetectedFace>& detectedFaces, const std::string& label)
{
    cv::CascadeClassifier faceCascade = cv::CascadeClassifier();
    if (!faceCascade.load(cascadeFilePath + Utils::SEP_CHAR + "haarcascade_frontalface_alt.xml")) {
        throw runtime_error("Error loading LBP cascade classifier");
    }

    cv::CascadeClassifier eyesCascade = cv::CascadeClassifier();
    if (!eyesCascade.load(cascadeFilePath + Utils::SEP_CHAR + "haarcascade_eye_tree_eyeglasses.xml")) {
        throw runtime_error("Error loading LBP cascade classifier");
    }
    cv::Mat gray;
    
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    vector<cv::Rect> faces;
    faceCascade.detectMultiScale(frame, faces, 1.1, 3, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20));

    if(faces.size() != 0) {
        for (size_t i = 0; i < faces.size(); i++) {
            cv::Mat faceROI = gray(faces[i]);
            size_t widthROI = faceROI.cols;
            size_t heightROI = faceROI.rows;
            float multiplier = float(faceROI.rows) / 480.;
            if(multiplier > 0.0) {
                widthROI /= multiplier;
                heightROI /= multiplier;
            }

            cv::Mat resizedFace;
            cv::resize(faceROI, resizedFace, cv::Size(widthROI, heightROI), cv::INTER_LINEAR);
            vector<cv::Rect> eyes;
            eyesCascade.detectMultiScale(resizedFace, eyes);
            if(eyes.size() != 0) {
                detectedFaces.push_back({faces[i], label});
            }
        }
    }
}

void FaceDetector::detectFacesWithLBP(const cv::Mat& frame, vector<DetectedFace>& detectedFaces)
{
    cv::CascadeClassifier lbpCascade = cv::CascadeClassifier();
    if (!lbpCascade.load(cascadeFilePath + Utils::SEP_CHAR + "lbpcascade_frontalcatface.xml")) {
        throw runtime_error("Error loading LBP cascade classifier");
    }

    cv::Mat gray;
    
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    vector<cv::Rect> faces;
    lbpCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    std::cout << "Faces found: " << faces.size() << endl;
    for (size_t i = 0; i < faces.size(); i++) {
        detectedFaces.push_back({faces[i], "unknown"});
    }
}