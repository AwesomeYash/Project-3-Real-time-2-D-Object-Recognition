/*
Project 3: Object Detector
Author: Priyanshu Ranka
NUID: 002305396
Date: 02/19/2025
Description: This project is a simple object detector that uses OpenCV to capture video from a camera and display it in a window.
The user can press 'q' to quit the program, 's' to save the current frame as an image, and any other key to apply a filter to the
video feed. The program will apply the selected filter to the video feed until the user selects a new filter or presses 's' to save
the current frame. The program will continue to run until the user presses 'q' to quit.
*/

// Include Directories
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <stdio.h>
#include <fstream>
#include "filters.h"

// Namespace
using namespace cv;
using namespace std;

// Define RegionFeatures struct
struct RegionFeatures
{
    double aspectRatio;
    double filledPercentage;
    std::vector<double> huMoments;
    double orientation; // Angle of the major axis
};

// Confusion Matrix (5x5 for five possible classes)
int confusionMatrix[5][5] = { 0 };
vector<string> possibleClasses = { "cup", "phone", "headphone", "remote", "butterfly" };

// Function to get class index
int getClassIndex(string className) {
    auto it = find(possibleClasses.begin(), possibleClasses.end(), className);
    return (it != possibleClasses.end()) ? distance(possibleClasses.begin(), it) : -1;
}

// Function to save debugging logs to a text file
void writeDebugLog(const string& filename, const string& logData) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "ERROR! Unable to open debug log file: " << filename << endl;
        return;
    }
    file << logData << endl;
    file.close();
}

// Function to classify using nearest neighbor
string classify(const vector<double>& inputFeature, const vector<string>& ClassName, const vector<vector<double>>& features) {
    if (features.empty()) return "Unknown";
    double minDistance = DBL_MAX;
    string bestMatch = "Unknown";
    for (size_t i = 0; i < features.size(); i++) {
        double distance = norm(inputFeature, features[i], NORM_L2);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = ClassName[i];
        }
    }
    writeDebugLog("debug_log.txt", "Predicted Class: " + bestMatch);
    return bestMatch;
}

// Function to print the confusion matrix
void printConfusionMatrix() {
    cout << "Confusion Matrix:" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << confusionMatrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Main function
int main(int, char**)
{
    cv::Mat frame;
    cv::VideoCapture cap;
    cv::Mat img;

    // Error handling
    cap.open(0);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    std::cout << "Start grabbing" << endl;
    int frameNumber = 0;
    int lastKeyPressed = 0;
    struct RegionFeatures rf;

    for (;;)
    {
        cap.read(frame);
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        Mat filteredFrame = frame.clone();
        filteredFrame = cleanImage(filteredFrame);
        Mat region_map = connectedComponentsAnalysis(filteredFrame);

        Mat Regions, stats, centroids;
        vector<int> CountTopLabels;
        Mat regionFrame = regions(filteredFrame, Regions, stats, centroids, CountTopLabels);

        vector<string> ClassName;
        vector<vector<double>> features;
        bool training = false;

        const char lastKeyPressed2 = waitKey(10);
        training = (lastKeyPressed2 == 't') ? !training : training;
        cout << (training ? "Training" : "Testing") << endl;

        loadFromCSV("./features.txt", ClassName, features);

        for (int n = 0; n < CountTopLabels.size(); n++)
        {
            int label = CountTopLabels[n];
            Mat region = (Regions == label);

            Moments m = moments(region, true);
            double CX = centroids.at<double>(label, 0);
            double CY = centroids.at<double>(label, 1);
            double theta = atan2(m.mu11, 0.5 * (m.mu20 - m.mu02));
            RotatedRect BoundingBox = obtainBoundingBox(region, CX, CY, theta);
            sketchLine(frame, CX, CY, theta, Scalar(0, 0, 255));
            sketchBoundingBox(frame, BoundingBox, Scalar(0, 255, 0));
            vector<double> hu_Moments;
            calculateHu_Moments(m, hu_Moments);

            string predictedClass = classify(hu_Moments, ClassName, features);
            writeDebugLog("debug_log.txt", "Hu Moments: " + to_string(hu_Moments[0]));

            putText(frame, predictedClass, Point(CX, CY), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        }

        cv::namedWindow("Original frame");
        cv::imshow("Original frame", frame);

        int keyPress = cv::waitKey(10);
        switch (keyPress)
        {
        case 'q':
            printConfusionMatrix();
            writeDebugLog("debug_log.txt", "Program exited");
            cap.release();
            destroyAllWindows();
            return 0;
        case 's':
            saveFrame(frame, ".", "frame" + to_string(frameNumber));
            break;
        default:
            lastKeyPressed = (keyPress != -1) ? keyPress : lastKeyPressed;
            break;
        }
        frameNumber++;
    }
    return 0;
}
