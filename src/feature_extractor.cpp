#include "feature_extractor.h"

using namespace std;

cv::Mat FeatureExtractor::extractFeatures(const cv::Mat& face, const cv::Mat& kCentres, int dictSize)
{
    // Extract features using SIFT
    cv::Mat descriptors;
    computeSIFTDescriptors(face, descriptors);

    return getDataVector(descriptors, kCentres, dictSize);    
}

cv::Mat FeatureExtractor::getDataVector(const cv::Mat& descriptors, const cv::Mat& kCenters, int dictSize)
{
    cv::BFMatcher matcher;
    vector<cv::DMatch> matches;
    matcher.match(descriptors, kCenters, matches);

    //Make a Histogram of visual words
    cv::Mat datai = cv::Mat::zeros(1, dictSize, CV_32F);
    int index = 0;
    for (auto j = matches.begin(); j < matches.end(); j++, index++) {
        datai.at<float>(0, matches.at(index).trainIdx) = datai.at<float>(0, matches.at(index).trainIdx) + 1;
    }

    return datai;
}

void FeatureExtractor::saveDescriptorsToFile(const string& filePath, const vector<cv::Mat>& descriptors)
{
    ofstream outFile(filePath);
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing: " << filePath << endl;
        return;
    }

    for (const auto& descriptor : descriptors) {
        outFile << "vector:" << endl;
        for (int i = 0; i < descriptor.rows; ++i) {
            for (int j = 0; j < descriptor.cols; ++j) {
                outFile << descriptor.at<float>(i, j);
                if (j < descriptor.cols - 1)
                    outFile << ",";
            }

            outFile << endl;
        }
    }

    outFile.close();
    std::cout << "Feature vectors saved to: " << filePath << endl;
}

vector<cv::Mat> FeatureExtractor::readDescriptorsFromFile(const string& filePath)
{
    vector<cv::Mat> descsPerImage;

    ifstream inFile(filePath);
    if (!inFile.is_open()) {
        cerr << "Error: Unable to open file for reading: " << filePath << endl;
        return descsPerImage;
    }

    string line;
    cv::Mat descriptors;
    while (getline(inFile, line)) {
        if (line == "vector:") {
            if (descriptors.rows > 0) {
                descsPerImage.push_back(descriptors);
                descriptors.release();
            }
        }
        else {        
            istringstream iss(line);
            string value;
            cv::Mat descRow(1, 0, CV_32F);
            while (getline(iss, value, ',')) {
                float valueNum;
                istringstream(value) >> valueNum;
                descRow.push_back(valueNum);
            }
            descRow = descRow.reshape(1, 1);

            descriptors.push_back(descRow);
        }
    }

    inFile.close();
    std::cout << "Feature vectors read from: " << filePath << endl;
    return descsPerImage;
}

cv::Mat FeatureExtractor::readDescriptorsFromMatFile(const string& filePath, const string& label)
{
    cv::Mat descriptors;

    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Unable to open file for reading: " << filePath << endl;
        return descriptors;
    }

    fs[label] >> descriptors;
    fs.release();

    if (descriptors.empty()) {
        cerr << "Error: Failed to read descriptors from file: " << filePath << endl;
    }

    return descriptors;
}

void FeatureExtractor::computeHOGDescriptors(const cv::Mat& face, vector<float>& descriptors)
{
    cv::Mat resizedFace;
    cv::resize(face, resizedFace, cv::Size(64, 128)); // HOG works best with this size
    cv::HOGDescriptor hog;
    hog.compute(face, descriptors);
}

void FeatureExtractor::computeSIFTDescriptors(const cv::Mat& face, cv::Mat& descriptors)
{
    cv::Mat resizedFace;
    cv::Mat grayFace;
    vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::SIFT> siftptr = cv::SIFT::create();

    cv::resize(face, resizedFace, cv::Size(64, 128));
    cv::cvtColor(resizedFace, grayFace, cv::COLOR_BGR2GRAY);

    siftptr->detectAndCompute(grayFace, cv::Mat(), keypoints, descriptors);
}