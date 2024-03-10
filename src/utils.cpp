#include "utils.h"

using namespace std;
namespace fs = filesystem;

vector<DetectedFace> Utils::readAnnotations(const string& filePath)
{
    vector<DetectedFace> annotations;

    ifstream inFile(filePath);
    if (!inFile.is_open()) {
        cerr << "Error: Unable to open file for reading: " << filePath << endl;
        return annotations;
    }

    string line;
    while (getline(inFile, line)) {
        istringstream iss(line);
        int x, y, width, height;
        string label;
        if (!(iss >> x >> y >> width >> height >> label)) {
            cerr << "Error: Invalid line format in file: " << filePath << endl;
            continue;
        }
        cv::Rect bbox(x, y, width, height);
        annotations.push_back(DetectedFace(bbox, label));
    }

    inFile.close();
    cout << "Annotations read from: " << filePath << endl;
    return annotations;
}

void Utils::writeAnnotations(const string& filePath, const vector<DetectedFace>& annotations)
{
    ofstream outFile(filePath);
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing: " << filePath << endl;
        return;
    }

    for (const auto& entry : annotations) {
        const cv::Rect& bbox = entry.getBoundingBox();
        outFile << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << " " << entry.getLabel() << endl;
    }

    outFile.close();
    cout << "Annotations saved to: " << filePath << endl;
}

void Utils::imagesToVideo(const vector<string>& imagePaths, const string& outputVideoPath, double fps)
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

vector<string> Utils::getImageFilesList(const string& path)
{
    vector<string> imagePaths;
    for (const auto & entry : fs::directory_iterator( path )) {
        auto fileExtension = string{entry.path().extension().string()};
        transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(),
                   [](unsigned char c) { return tolower( c ); });

        if (fileExtension == ".png" || fileExtension == ".jpg" || fileExtension == ".jpeg")
            imagePaths.push_back(entry.path().string());
    }

    sort(imagePaths.begin(), imagePaths.end(), [](const string& str1, const string& str2) {
        string name1 = str1.substr(str1.rfind(Utils::SEP_CHAR) + 1);
        string name2 = str2.substr(str2.rfind(Utils::SEP_CHAR) + 1);
        name1 = name1.substr(0, name1.rfind('.'));
        name2 = name2.substr(0, name2.rfind('.'));
        int num1 = 0, num2 = 0;
        try {
            num1 = stoi(name1);
            num2 = stoi(name2);
        } catch(...) {
            cerr << "Failed to convert " << name1 << "/" << name2 << " to integer" <<endl;
        }

        return num1 < num2;
    });

    return imagePaths;
}

vector<string> Utils::getVideosFilesList(const std::string& path)
{
    vector<string> videoPaths;
    for (const auto & entry : fs::directory_iterator( path )) {
        auto fileExtension = string{entry.path().extension().string()};
        transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(),
                   [](unsigned char c) { return tolower( c ); });

        if (fileExtension == ".avi" || fileExtension == ".mp4" || fileExtension == ".mkv")
            videoPaths.push_back(entry.path().string());
    }

    sort(videoPaths.begin(), videoPaths.end(), [](const string& str1, const string& str2) {
        string name1 = str1.substr(str1.rfind(Utils::SEP_CHAR) + 1);
        string name2 = str2.substr(str2.rfind(Utils::SEP_CHAR) + 1);
        name1 = name1.substr(0, name1.rfind('.'));
        name2 = name2.substr(0, name2.rfind('.'));
        int num1 = 0, num2 = 0;
        try {
            num1 = stoi(name1);
            num2 = stoi(name2);
        } catch(...) {
            cerr << "Failed to convert " << name1 << "/" << name2 << " to integer" <<endl;
        }

        return num1 < num2;
    });

    return videoPaths;
}

vector<string> Utils::getDirectoriesList(const string& path)
{
    vector<string> dirPaths;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_directory()) {
            dirPaths.push_back(entry.path().string());
        }
    }

    return dirPaths;
}

void Utils::convertAllTestingVideos(const std::string& path)
{
    auto dirPaths = getDirectoriesList(path);
    for (const auto& dirPath : dirPaths) {
        auto innerDirs = getDirectoriesList(dirPath);
        auto subjectName = dirPath.substr(dirPath.rfind(SEP_CHAR) + 1);
        for (const auto& innerDir : innerDirs) {
            std::stringstream ss;
            auto num = innerDir.substr(innerDir.rfind(SEP_CHAR) + 1);
            ss << dirPath << SEP_CHAR << subjectName << "_" << num << ".avi";
            auto imagePaths = getImageFilesList(innerDir);
            imagesToVideo(imagePaths, ss.str());
            ss.str("");
        }
    }
}