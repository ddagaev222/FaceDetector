#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include <algorithm>

#include "detected_face.h"

class Utils
{
public:
    // Method to read annotations from a text file
    static std::vector<DetectedFace> readAnnotations(const std::string& filePath);

    // Method to write annotations to a text file
    static void writeAnnotations(const std::string& filePath, const std::vector<DetectedFace>& annotations);

    // Method to convert set of images to video file
    static void imagesToVideo(const std::vector<std::string>& imagePaths, const std::string& outputVideoPath, double fps = 30);

    // Method to get list of all images in the directory
    static std::vector<std::string> getImageFilesList(const std::string& path);

    // Method to get list of all videos in the directory
    static std::vector<std::string> getVideosFilesList(const std::string& path);

    // Method to get list of all directories in the given path
    static std::vector<std::string> getDirectoriesList(const std::string& path);

    // Method to check if file exists
    static inline bool fileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    // Method to convert all sets of images into videos for testing
    static void convertAllTestingVideos(const std::string& path);

public:
    static const char SEP_CHAR = std::filesystem::path::preferred_separator;

};

#endif // UTILS_H
