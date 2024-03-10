#include "annotation.h"

using namespace std;

vector<DetectedFace> Annotation::readAnnotations(const string& filePath)
{
    vector<DetectedFace> annotations;

    ifstream inFile(filePath);
    if (!inFile.is_open()) {
        cerr << "Error: Unable to open file for reading: " << filePath << endl;
        return annotations;
    }

    auto lastSepPos = filePath.rfind(Utils::SEP_CHAR);
    auto labelStr = filePath.substr(0, lastSepPos);
    labelStr = labelStr.substr(labelStr.rfind(Utils::SEP_CHAR) + 1);

    string line;
    while(getline(inFile, line)) {
        istringstream lineStream(line);
        string filename, strPart;
        int x, y, width, height;
        try {
            getline( lineStream, strPart, ',' );
            filename = strPart;
            getline( lineStream, strPart, ',' );
            getline( lineStream, strPart, ',' );
            x = stoi( strPart );
            getline( lineStream, strPart, ',' );
            y = stoi( strPart );
            getline( lineStream, strPart, ',' );
            width = stoi( strPart );
            getline( lineStream, strPart, ',' );
            height = stoi( strPart );
        } catch(...) {
            continue;
        }
        
        cv::Rect faceRect{x - width / 2, y - height / 2, width, height};
        annotations.push_back(DetectedFace{faceRect, labelStr});
    }

    inFile.close();
    return annotations;
}

void Annotation::imagesToVideo(const vector<string>& imagePaths, const string& outputVideoPath, double fps)
{
    cv::Size frameSize;
    cv::Mat frame;

    // Load the first image to get its size
    frame = cv::imread(imagePaths[0]);
    if (frame.empty()) {
        cerr << "Failed to load image: " << imagePaths[0] << endl;
        return;
    }
    frameSize = frame.size();

    // Video writer object
    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, frameSize);

    // Loop through each image and add it to the video
    for (const auto& imagePath : imagePaths) {
        frame = cv::imread(imagePath);
        if (frame.empty()) {
            cerr << "Failed to load image: " << imagePath << endl;
            continue;
        }
        videoWriter.write(frame);
    }

    // Release the video writer
    videoWriter.release();
}