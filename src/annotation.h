#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "detected_face.h"
#include "utils.h"

class Annotation
{
public:
    // Method to read annotations from a CSV file
    static std::vector<DetectedFace> readAnnotations(const std::string& filePath);

    // Method to convert set of images to video file
    static void imagesToVideo(const std::vector<std::string>& imagePaths, const std::string& outputVideoPath, double fps);

};

#endif // ANNOTATION_H
