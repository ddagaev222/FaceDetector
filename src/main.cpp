#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "utils.h"
#include "face_recognizer.h"

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

string train_path;
string test_path;
bool test = false;
bool compute = false;

void printUsage(char** argv)
{
    cout <<
        "Usage:\n" << argv[0] << " [Flags] \n"
        "Flags:\n"
        "  -mode (test|live)\n"
        "  -desc (read|compute)\n"
        "\n"
        "Examples:\n"
        "-mode test -desc read -train_path C:\\Projects\\face_detector\\train_data -test_path C:\\Projects\\face_detector\\test_data\n"
        "-mode live -desc compute -train_path /home/Projects/face_detector/train_data -test_path /home/Projects/face_detector/test_data\n";
}

int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return EXIT_FAILURE;
        }
        else if (string(argv[i]) == "-mode")
        {
            if (string(argv[i + 1]) == "test")
            {
                test = true;
            }
            else if (string(argv[i + 1]) == "live")
            {
                test = false;
            }
            else
            {
                cout << "Bad -mode flag value" << endl;
                return EXIT_FAILURE;
            }

            i++;
        }
        else if (string(argv[i]) == "-desc")
        {
            if (string(argv[i + 1]) == "compute")
            {
                compute = true;
            }
            else if (string(argv[i + 1]) == "read")
            {
                compute = false;
            }
            else
            {
                cout << "Bad -desc flag value" << endl;
                return EXIT_FAILURE;
            }

            i++;
        }
        else if (string(argv[i]) == "-train_path")
        {
            if (!string(argv[i + 1]).empty()) {
                train_path = argv[i + 1];
            }
            else
            {
                cout << "Bad -img flag value" << endl;
                return EXIT_FAILURE;
            }

            i++;
        }
        else if (string(argv[i]) == "-test_path")
        {
            if (!string(argv[i + 1]).empty()) {
                test_path = argv[i + 1];
            }
            else
            {
                cout << "Bad -img flag value" << endl;
                return EXIT_FAILURE;
            }
        }
    }

    if (train_path.empty()) {
        cout << "Please, provide the path to train dataset on your device" << endl;
        return EXIT_FAILURE;
    }

    if (test && test_path.empty()) {
        cout << "To perform test mode, please, provide the path to test dataset on your device" << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    if (parseCmdArgs(argc, argv))
        return EXIT_FAILURE;

    // The following instruction convert images from YouTube Faces Dataset to videos for testing
    // Utils::convertAllTestingVideos("C:\\Projects\\face_detector\\faces");

    // FaceRecognizer faceRecognizerkNN(TRAIN_DATA_PATH, TRAIN_DATA_PATH);
    // Compute descriptors from training images and save them to CSV files, then execute features and train model and print quality metrics
    // faceRecognizerkNN.train(5, 1e4, 10, DICT_SIZE, true, false, true, "descriptors.csv", true, "annotation.csv");

    // Read descriptors from CSV files and train model
    // faceRecognizerkNN.train(DICT_SIZE, 5, 1e4, 10, false);

    string answer;
    string modelType = "knn", detectionAlgorithm = "haar", descriptorsAlgorithm = "sift";
    string imagesNumStr, attemptsNumStr, dictSizeStr, iterNumStr, kStr;
    string showTrainStr, showTestStr, saveDescStr, showQMStr, thresholdStr;
    string descriptorsFilename = "descriptors.csv", annotationFilename = "annotation.csv";

    int imagesPerSubj, attempts, dictSize, iterNum, k, knnThreshold;
    bool showTrain, showTest, saveDesc, qm;

    if (test) {
        imagesPerSubj = 10;
        attempts = 5;
        dictSize = 230;
        iterNum = 1e4;
        k = 5;
        knnThreshold = 49;
        showTrain = false;
        showTest = true;
        saveDesc = true;
        qm = true;
    } else {
        imagesPerSubj = 10;
        attempts = 5;
        dictSize = 230;
        iterNum = 1e4;
        k = 5;
        knnThreshold = 48;
        showTrain = true;
        showTest = false;
        saveDesc = true;
        qm = false;
    }

    cout << "You are welcome to FaceControl app based on YouTube Faces dataset!\nFirst read our rules:" << endl;
    cout << "Make sure that the following files are in the training data folder:\n"
         << "- haarcascade_eye_tree_eyeglasses.xml\n- haarcascade_frontalface_alt.xml\n- lbpcascade_frontalcatface.xml\n"
         << "They are vital for the correct operation of the application."
         << "Do you want to configure some algorithms' parameters? The default parameters are optimized.(Y/N)" << endl;
    cin >> answer;
    transform(answer.begin(), answer.end(), answer.begin(),
              [](unsigned char c) { return tolower( c ); });
    if (answer == "y" || answer == "yes") {
        cout << "I can't handle errors, so be careful with your inputs;)" << endl;
        cout << "Model (kNN | SVM): ";
        cin >> modelType;
        cout << "Face detection algorithm (Haar | LBP): ";
        cin >> descriptorsAlgorithm;
        cout << "Descriptors algorithm (SIFT | HOG): ";
        cin >> descriptorsAlgorithm;
        cout << "Train images number per subject(10): ";
        cin >> imagesNumStr;
        cout << "Attempts for training(5): ";
        cin >> attemptsNumStr;
        cout << "Dict size(230): ";
        cin >> dictSizeStr;
        cout << "K Neighbours: ";
        cin >> kStr;
        cout << "kNN threshold: ";
        cin >> thresholdStr;
        cout << "Number of training iterations(10000): ";
        cin >> iterNumStr;
        cout << "Show train detection results(Y/N): ";
        cin >> showTrainStr;
        cout << "Show test detection results(Y/N): ";
        cin >> showTestStr;
        cout << "Save descriptors to file(Y/N): ";
        cin >> saveDescStr;
        cout << "Show quality metrics on train(Y/N): ";
        cin >> showQMStr;
        cout << "Descriptors filename(descriptors.csv): ";
        cin >> descriptorsFilename;
        cout << "Annotation filename(annotation.csv): ";
        cin >> annotationFilename;

        try {
            imagesPerSubj = stoi(imagesNumStr);
            attempts = stoi(attemptsNumStr);
            dictSize = stoi(dictSizeStr);
            iterNum = stoi(iterNumStr);
            k = stoi(kStr);
            knnThreshold = stoi(thresholdStr);
            transform(showTrainStr.begin(), showTrainStr.end(), showTrainStr.begin(),
                      [](unsigned char c) { return tolower( c ); });
            if (showTrainStr == "y" || showTrainStr == "yes")
                showTrain = true;
            else
                showTrain = false;
            transform(showTestStr.begin(), showTestStr.end(), showTestStr.begin(),
                      [](unsigned char c) { return tolower( c ); });
            if (showTestStr == "y" || showTestStr == "yes")
                showTest = true;
            else
                showTest = false;
            transform(saveDescStr.begin(), saveDescStr.end(), saveDescStr.begin(),
                      [](unsigned char c) { return tolower( c ); });
            if (saveDescStr == "y" || saveDescStr == "yes")
                saveDesc = true;
            else
                saveDesc = false;
            transform(showQMStr.begin(), showQMStr.end(), showQMStr.begin(),
                      [](unsigned char c) { return tolower( c ); });
            if (showQMStr == "y" || showQMStr == "yes")
                qm = true;
            else
                qm = false;
        } catch(...) {
            cerr << "Oops, something went wrong, try again later! You can setup the parameters in the code as you want.\nGoodbye:)" << endl;
            return EXIT_FAILURE;
        }
    }

    cout << "Let's GO!" << endl;

    // Create recognizer and train
    FaceRecognizer faceRecognizer(train_path, train_path, modelType, detectionAlgorithm, descriptorsAlgorithm, dictSize);
    faceRecognizer.train(attempts, iterNum, imagesPerSubj, compute, showTrain, saveDesc, descriptorsFilename, qm, annotationFilename);

    if (test) {
        // Test the model on videos
        faceRecognizer.runTest(test_path, k, knnThreshold);
    } else {
        // Recognize Faces on Live Cam
        faceRecognizer.runLive(k, knnThreshold);
    }

    return EXIT_SUCCESS;
}
