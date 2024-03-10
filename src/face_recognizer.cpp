#include "face_recognizer.h"

using namespace std;

FaceRecognizer::FaceRecognizer(const string& trainDataPath, const string& cascadeFilePath,
                               const string& modelType, const string& detectionAlgorithm,
                               const string& descriptorsAlgorithm, int dictSize)
    : trainDataPath{trainDataPath}
    , modelType{modelType}
    , pFaceDetector{cv::makePtr<FaceDetector>(cascadeFilePath, detectionAlgorithm)}
    , pTracker{cv::makePtr<FaceTracker>()}
    , descriptorsAlgorithm{descriptorsAlgorithm}
    , dictSize{dictSize}
    , trained{false}
{
    if (modelType == "knn") {
        pModel = cv::ml::KNearest::create();
    } else {
        pModel = cv::ml::SVM::create();
    }
}

void FaceRecognizer::train(int attempts, int iterationNumber, int trainImagesNumber, bool compute, bool show,
                           bool save, const string& descFilename, bool quality, const string& annotFilename)
{
    cv::Mat allDescriptors;
    vector<cv::Mat> allDescPerImg;
    vector<int> allClassPerImg;
    int allDescPerImgNum = 0;

    std::cout << "****** Preparing descriptors ******" << endl;
    if (compute) {
        if (quality)
            std::cout << "****** Reading annotations and computing metrics ******" << endl;
        
        computeAndPrepareDescriptors(trainImagesNumber, allDescriptors, allClassNames, allDescPerImg,
                                     allClassPerImg, allDescPerImgNum, show, save, descFilename, quality, annotFilename);
    } else {
        readDescriptors(descFilename, allDescriptors, allClassNames, allDescPerImg, allClassPerImg, allDescPerImgNum);
    }

    std::cout << "****** Clustering ******" << endl;
    cv::Mat kLabels;
    cv::kmeans(allDescriptors, dictSize, kLabels,
               cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, iterationNumber, 1e-4),
               attempts, cv::KMEANS_PP_CENTERS, kCenters);

    std::cout << "****** Histogram ******" << endl;
    cv::Mat inputData;
    cv::Mat inputDataLables;
    for (int i = 0; i < allDescPerImgNum; i++) {
        cv::Mat datai = FeatureExtractor::getDataVector(allDescPerImg[i], kCenters, dictSize);

        inputData.push_back(datai);
        inputDataLables.push_back(cv::Mat(1, 1, CV_32SC1, allClassPerImg[i]));
    }

    std::cout << "****** Training model ******" << endl;
    cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(inputData, cv::ml::ROW_SAMPLE, inputDataLables);
    if (modelType == "svm") {
        cv::Ptr<cv::ml::SVM> svmPtr = pModel.dynamicCast<cv::ml::SVM>();
        svmPtr->setType(cv::ml::SVM::C_SVC);
        svmPtr->setKernel(cv::ml::SVM::LINEAR);
        svmPtr->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, attempts, 1e-6));
        svmPtr->train(td);
        trained = true;
    }
    else if (modelType == "knn") {
        pModel->train(td);
        trained = true;
    }

    if (quality) {
        std::cout << "****** Final training metrics ******" << endl;
        computeMeanMetricsAndPrint(trainMetrics);
    }
}

void FaceRecognizer::runLive(int k, float thresholdDistance)
{
    // Create a VideoCapture object to capture video from the default camera (index 0)
    cv::VideoCapture videoCapture(0);

    // Check if the camera is opened successfully
    if (!videoCapture.isOpened()) {
        cerr << "Error: Failed to open camera." << std::endl;
        return;
    }

    // Create a window to display the captured video
    cv::namedWindow("LiveCam", cv::WINDOW_AUTOSIZE);

    // Initialize variables for face detection
    std::vector<DetectedFace> detectedFaces;
    bool performDetection = true;

    // Process video frames
    cv::Mat frame;
    int frameCount = 0;

    while (true) {
        // Capture frame-by-frame
        videoCapture >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame." << std::endl;
            break;
        }

        if (performDetection) {
            pTracker->dropTrackers();
            detectedFaces = pFaceDetector->detectFaces(frame);
            if (detectedFaces.empty()) {
                putText(frame, "Cannot detect faces", cv::Point(10, 10), fontText, textScale, cv::Scalar(0, 0, 255), thickness);
                
                cv::imshow("LiveCam", frame);
                int keyboard = cv::waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;

                continue;
            }
            for (auto& face : detectedFaces) {
                cv::Rect faceRect = face.getBoundingBox();
                if (!faceRect.empty()) {
                    cv::Mat FaceROI = frame(faceRect);
                    cv::Mat descriptors;
                    FeatureExtractor::computeSIFTDescriptors(FaceROI, descriptors);
                    cv::Mat dvector = FeatureExtractor::getDataVector(descriptors, kCenters, dictSize);
                    if (modelType == "knn") {
                        cv::Mat results, neighbors, distances;
                        cv::Ptr<cv::ml::KNearest> kNNPtr = pModel.dynamicCast<cv::ml::KNearest>();
                        kNNPtr->setDefaultK(k);
                        kNNPtr->setIsClassifier(true);
                        
                        // Predict nearest neighbours
                        kNNPtr->findNearest(dvector, k, results, neighbors, distances);

                        // Screen out distant neighbors and mark them as unknown
                        std::vector<int> unknown_indices;
                        for (int i = 0; i < distances.rows; ++i) {
                            if (distances.at<float>(i, 0) > thresholdDistance) {
                                unknown_indices.push_back(i);
                            }
                        }
                        
                        for (int i = 0; i < results.rows; ++i) {
                            if (std::find(unknown_indices.begin(), unknown_indices.end(), i) != unknown_indices.end()) {
                                face.setLabel("Unknown");
                            } else {
                                face.setLabel(allClassNames[results.at<float>(i, 0)]);
                            }
                        }
                    } else {
                        float prediction = pModel->predict(dvector);
                        face.setLabel(allClassNames[prediction]);
                    }
                }
            }
            performDetection = false;
        }

        // Track detected faces in each frame
        pTracker->trackFaces(frame, detectedFaces);

        for (auto& face : detectedFaces) {
            cv::Rect faceRect = face.getBoundingBox();
            if (!faceRect.empty()) {
                rectangle(frame, faceRect, cv::Scalar(255, 0, 0), 2, 1);
                string textLabel = face.getLabel();
                cv::Point textPoint(faceRect.x, faceRect.y - 5);
                cv::Size textSize = cv::getTextSize(textLabel, fontText, textScale, thickness, &baseLine);
                computeTextPosition(textPoint, textSize, frame.rows, frame.cols);
                putText(frame, textLabel, textPoint, fontText, textScale, textColor, thickness);
            }
        }

        // Increment frame count and reset performDetection flag
        frameCount++;
        if (frameCount % 10 == 0)
            performDetection = true;

        // Display the captured frame
        cv::imshow("LiveCam", frame);
        int keyboard = cv::waitKey(10);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }

    // Release the VideoCapture object and close the window
    videoCapture.release();
    cv::destroyAllWindows();

}

void FaceRecognizer::runTest(const string& testPath, bool show, int k, float thresholdDistance, const string& annotationFilename)
{
    if (trained) {
        std::cout << "****** Testing ******" << endl;
        auto dirPaths = Utils::getDirectoriesList(testPath);
        for (const auto& dirPath : dirPaths) {
            testPerClass(dirPath, show, k, thresholdDistance, annotationFilename);
        }
    } else {
        cerr << "Can't run Test until FaceRecognizer is trained.";
        return;
    }
    std::cout << "****** Final test metrics ******" << endl;
    computeMeanMetricsAndPrint(testMetrics);
}

void FaceRecognizer::testPerClass(const string& classPath, bool show, int k, float thresholdDistance, const string& annotationFilename)
{
    vector<vector<DetectedFace>> allDetectedFaces;
    auto videoFiles = Utils::getVideosFilesList(classPath);
    auto className = classPath.substr(classPath.rfind(Utils::SEP_CHAR) + 1);
    cv::namedWindow("Test_" + className);

    for (const auto& filename : videoFiles) {
        cv::VideoCapture videoCapture(filename);

        // Check if the video capture is open
        if (!videoCapture.isOpened()) {
            throw runtime_error("Video capture is not open");
        }

        // Initialize variables for face detection
        bool performDetection = true;

        // Process video frames
        cv::Mat frame;
        int frameCount = 0;
        while (videoCapture.read(frame)) {
            // Perform face detection once per 10 frames
            if (performDetection) {
                pTracker->dropTrackers();
                auto detectedFaces = pFaceDetector->detectFaces(frame);
                if (detectedFaces.empty()) {
                    putText(frame, "Cannot detect faces", cv::Point(10, 10), fontText, textScale, cv::Scalar(0, 0, 255), thickness);
                    
                    if (show) {
                        cv::imshow("Test_" + className, frame);
                        int keyboard = cv::waitKey(30);
                        if (keyboard == 'q' || keyboard == 27)
                            break;
                    }

                    continue;
                }
                for (auto& face : detectedFaces) {
                    cv::Rect faceRect = face.getBoundingBox();
                    if (!faceRect.empty()) {
                        cv::Mat FaceROI = frame(faceRect);
                        cv::Mat descriptors;
                        FeatureExtractor::computeSIFTDescriptors(FaceROI, descriptors);
                        cv::Mat dvector = FeatureExtractor::getDataVector(descriptors, kCenters, dictSize);
                        if (modelType == "knn") {
                            cv::Mat results, neighbors, distances;
                            cv::Ptr<cv::ml::KNearest> kNNPtr = pModel.dynamicCast<cv::ml::KNearest>();
                            kNNPtr->setDefaultK(k);
                            kNNPtr->setIsClassifier(true);
                            
                            // Predict nearest neighbours
                            kNNPtr->findNearest(dvector, k, results, neighbors, distances);

                            // Screen out distant neighbors and mark them as unknown
                            std::vector<int> unknown_indices;
                            for (int i = 0; i < distances.rows; ++i) {
                                if (distances.at<float>(i, 0) > thresholdDistance) {
                                    unknown_indices.push_back(i);
                                }
                            }
                            
                            for (int i = 0; i < results.rows; ++i) {
                                if (std::find(unknown_indices.begin(), unknown_indices.end(), i) != unknown_indices.end()) {
                                    face.setLabel("Unknown");
                                } else {
                                    face.setLabel(allClassNames[results.at<float>(i, 0)]);
                                }
                            }
                        } else {
                            float prediction = pModel->predict(dvector);
                            face.setLabel(allClassNames[prediction]);
                        }
                    }
                }
                allDetectedFaces.push_back(detectedFaces);
                performDetection = false;
            }

            auto detectedFaces = allDetectedFaces.back();
            // Track detected faces in each frame
            pTracker->trackFaces(frame, detectedFaces);
            allDetectedFaces.push_back(detectedFaces);
            for (auto& face : detectedFaces) {
                cv::Rect faceRect = face.getBoundingBox();
                if (!faceRect.empty()) {
                    rectangle(frame, faceRect, cv::Scalar(255, 0, 0), 2, 1);
                    string textLabel = face.getLabel();
                    cv::Point textPoint(faceRect.x, faceRect.y - 5);
                    cv::Size textSize = cv::getTextSize(textLabel, fontText, textScale, thickness, &baseLine);
                    computeTextPosition(textPoint, textSize, frame.rows, frame.cols);
                    putText(frame, textLabel, textPoint, fontText, textScale, textColor, thickness);
                }
            }

            // Increment frame count and reset performDetection flag
            frameCount++;
            if (frameCount % 10 == 0)
                performDetection = true;

            if (show) {
            cv::imshow("Test_" + className, frame);
                int keyboard = cv::waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;
            }
        }
        videoCapture.release();
        cv::destroyWindow("Test_" + className);
    }

    if (className != "Unknown") {
        auto annotatedFaces = Annotation::readAnnotations(classPath + Utils::SEP_CHAR + annotationFilename);
        QualityMetrics metrics;
        computeMetrics(allDetectedFaces, annotatedFaces, metrics);
        printMetrics(metrics);
        testMetrics.push_back(metrics);
    }
}

void FaceRecognizer::computeTrainDescriptors(const string& className, int imagesNumber, int classLabel, cv::Mat& allDescriptors,
                                             vector<cv::Mat>& allDescPerImg, vector<int>& allClassPerImg, int& allDescPerImgNum, bool showResults,
                                             bool saveToFile, const string& descFilename, bool quality, const string& annotationsFilename)
{
    string classPath{trainDataPath + Utils::SEP_CHAR + className};
    vector<string> imagePaths = Utils::getImageFilesList(classPath);
    vector<cv::Mat> descsPerClass;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    cv::namedWindow("Train");

    vector<vector<DetectedFace>> allDetectedFaces;
    int imagesToRead = imagesNumber > imagePaths.size() ? imagePaths.size() : imagesNumber;
    for (int i = 0; i < imagesToRead; i++) {
        cv::Mat image = cv::imread(imagePaths[i]);
        vector<DetectedFace> detectedFaces = pFaceDetector->detectFaces(image, className);
        
        if (detectedFaces.empty()) {
            cout << "Can't detect any faces on " << imagePaths[i] << endl;
            continue;
        }

        if (detectedFaces.size() > 1) {
            cout << imagePaths[i] << ": " << detectedFaces.size() << " faces found" << endl;
        }

        if (quality)
            allDetectedFaces.push_back(detectedFaces);
        
        for (const auto& face : detectedFaces) {
            cv::Rect faceRect = face.getBoundingBox();
            if (!faceRect.empty()) {
                string textLabel = face.getLabel();
                rectangle(image, faceRect, cv::Scalar(255, 0, 0), 2, 1);
                cv::Point textPoint(faceRect.x, faceRect.y - 5);
                cv::Size textSize = cv::getTextSize(textLabel, fontText, textScale, thickness, &baseLine);
                computeTextPosition(textPoint, textSize, image.rows, image.cols);
                putText(image, textLabel, textPoint, fontText, textScale, textColor, thickness);

                cv::Mat FaceROI = image(faceRect);
                FeatureExtractor::computeSIFTDescriptors(FaceROI, descriptors);
                allDescriptors.push_back(descriptors);
                allDescPerImg.push_back(descriptors);
                allClassPerImg.push_back(classLabel);
                allDescPerImgNum++;
            }
        }

        if (showResults) {
            cv::imshow("Train", image);

            int keyboard = cv::waitKey(0);
            if (keyboard == 'q' || keyboard == 27)
                showResults = false;
        }

        if (saveToFile)
            descsPerClass.push_back(descriptors);
    }

    cv::destroyWindow("Train");

    if (quality) {
        auto annotatedFaces = Annotation::readAnnotations(classPath + Utils::SEP_CHAR + annotationsFilename);
        QualityMetrics metrics;
        computeMetrics(allDetectedFaces, annotatedFaces, metrics);
        printMetrics(metrics);
        trainMetrics.push_back(metrics);
    }

    if (saveToFile)
        FeatureExtractor::saveDescriptorsToFile(classPath + Utils::SEP_CHAR + descFilename, descsPerClass);
}

void FaceRecognizer::computeAndPrepareDescriptors(int imagesNumber, cv::Mat& allDescriptors, vector<string>& allClassNames,
                                                  vector<cv::Mat>& allDescPerImg, vector<int>& allClassPerImg, int& allDescPerImgNum, bool showResults,
                                                  bool saveToFile, const string& descFilename, bool quality, const string& annotFilename)
{
    auto dirPaths = Utils::getDirectoriesList(trainDataPath);
    int classLabel = 0;
    for (const auto& dirPath : dirPaths) {
        auto tokenPos = dirPath.rfind(Utils::SEP_CHAR);
        auto className = dirPath.substr(tokenPos + 1);
        computeTrainDescriptors(className, imagesNumber, classLabel, allDescriptors, allDescPerImg, allClassPerImg,
                                allDescPerImgNum, showResults, saveToFile, descFilename, quality, annotFilename);
        allClassNames.push_back(className);
        allClassLabels.push_back(classLabel);
        classLabel++;
    }
}

void FaceRecognizer::readDescriptors(const string& filename, cv::Mat& allDescriptors, vector<string>& allClassNames,
                                     vector<cv::Mat>& allDescPerImg, vector<int>& allClassPerImg, int& allDescPerImgNum)
{
    auto dirPaths = Utils::getDirectoriesList(trainDataPath);
    int classLabel = 0;
    for (const auto& dirPath : dirPaths) {
        auto tokenPos = dirPath.rfind(Utils::SEP_CHAR);
        auto className = dirPath.substr(tokenPos + 1);
        vector<cv::Mat> descriptors = FeatureExtractor::readDescriptorsFromFile(dirPath + Utils::SEP_CHAR + filename);
        allClassNames.push_back(className);
        for (int i = 0; i < descriptors.size(); ++i) {
            allDescriptors.push_back(descriptors[i]);
            allDescPerImg.push_back(descriptors[i]);
            allClassPerImg.push_back(classLabel);
        }
        allDescPerImgNum += descriptors.size();
        allClassLabels.push_back(classLabel);
        classLabel++;
    }
}

double FaceRecognizer::computeDistance(const cv::Rect& rect1, const cv::Rect& rect2)
{
    cv::Point center1(rect1.x + rect1.width / 2, rect1.y + rect1.height / 2);
    cv::Point center2(rect2.x + rect2.width / 2, rect2.y + rect2.height / 2);

    return cv::norm(center1 - center2);
}

void FaceRecognizer::computeTextPosition(cv::Point& textPoint, const cv::Size& textSize, int rows, int cols)
{
    if (textPoint.x < 0)
        textPoint.x = 3;
    else if (textPoint.x + textSize.width > cols)
        textPoint.x = cols - textSize.width - 3;

    if (textPoint.y - textSize.height < 0)
        textPoint.y = textSize.height + 3;
    else if (textPoint.y > rows)
        textPoint.y = rows - 3;
}

void FaceRecognizer::computeMetrics(const vector<vector<DetectedFace>> foundFaces, const vector<DetectedFace> trueFaces, QualityMetrics& metrics, double eps)
{
    int facesNumDiff = foundFaces.size() - trueFaces.size();
    int maxFacesNum = foundFaces.size();
    int correctPredictedFaces = 0;
    if (facesNumDiff > 0) {
        maxFacesNum -= facesNumDiff;
        metrics.mFP += facesNumDiff;
    } else if (facesNumDiff < 0) {
        metrics.mFN -= facesNumDiff;
    }

    metrics.elementsCount = maxFacesNum;

    std::vector<double> distancesVector;
    for (size_t i = 0; i < maxFacesNum; ++i) {
        for (size_t j = 0; j < foundFaces[i].size(); ++j) {
            if (foundFaces[i][j].getBoundingBox().empty()) {
                metrics.mFN++;
            } else {
                double distance = computeDistance(foundFaces[i][j].getBoundingBox(), trueFaces[i].getBoundingBox());
                distancesVector.push_back(distance);
                if (distance < eps) {
                    metrics.mTP++;
                } else {
                    metrics.mFP++;
                }
                if (foundFaces[i][j].getLabel() == trueFaces[i].getLabel())
                    correctPredictedFaces++;
            }
        }
    }
    
    if (!distancesVector.empty()) {
        int sum = accumulate(distancesVector.begin(), distancesVector.end(), 0);
        metrics.mDetectionError = static_cast<double>(sum) / distancesVector.size();
    }

    metrics.mAccuracy = static_cast<double>(correctPredictedFaces) / trueFaces.size();
}

void FaceRecognizer::printMetrics(const QualityMetrics& metrics) const
{
    cout << "Dataset volume: " << metrics.elementsCount << endl;
    cout << "True Positives: " << metrics.mTP << endl;
    cout << "False Positives: " << metrics.mFP << endl;
    cout << "False Negatives: " << metrics.mFN << endl;
    cout << "Mean Detection Error: " << metrics.mDetectionError << " in pixels" << endl;
    cout << "Accuracy: " << metrics.mAccuracy << endl;
}

void FaceRecognizer::computeMeanMetricsAndPrint(const vector<QualityMetrics>& metrics) const
{
    int datasetSize = 0, sumTP = 0, sumFP = 0, sumFN = 0;
    double sumDetectionError = 0.0, sumAccuracy = 0.0;

    for (const auto& metric : metrics) {
        datasetSize += metric.elementsCount;
        sumTP += metric.mTP;
        sumFP += metric.mFP;
        sumFN += metric.mFN;
        sumDetectionError += metric.mDetectionError;
        sumAccuracy += metric.mAccuracy;
    }

    QualityMetrics meanMetrics{datasetSize, sumTP, sumFP, sumFN, sumDetectionError / metrics.size(), sumAccuracy / metrics.size()};

    printMetrics(meanMetrics);
}