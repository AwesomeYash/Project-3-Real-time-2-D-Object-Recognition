#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

string saveFrame(cv::Mat& frame, std::string path, std::string filename);

// Task 1: Thresholding function
cv::Mat thresholdImage(cv::Mat& image);

// Task 2: Cleaning the image
cv::Mat cleanImage(cv::Mat& image);

// Dilation of image
cv::Mat dilateImage(cv::Mat inputImage, cv::Size kernelSize, cv::Point anchor, int iterations);

// Erosion of image
cv::Mat erodeImage(cv::Mat inputImage, cv::Size kernelSize, cv::Point anchor, int iterations);

// Task 3: Region Segmentation
// Function to generate random colors
vector<cv::Vec3b> generateRandomColors(int num_colors);

// Task 4: Region Features
Mat regions(Mat& image, Mat& labeledRegions, Mat& stats, Mat& centroids, vector<int>& topNLabels);

// Method that calculates the bounding box
RotatedRect obtainBoundingBox(Mat& region, double x, double y, double theta);

// Method that draws the axis line
void sketchLine(Mat& image, double x, double y, double theta, Scalar color);

// Method that draws the bounding box
void sketchBoundingBox(Mat& image, RotatedRect boundingBox, Scalar color);

// Method that calculates the features
void calculateHu_Moments(Moments mo, vector<double>& hu_Moments);

// Method that calculates the Euclidean Distance
double computeEuclideanDistance(vector<double> feats1, vector<double> feats2);

// Connected Components Analysis
Mat connectedComponentsAnalysis(const cv::Mat& thresholded_image);

// Method that compares the features of the frame withe stored data
string classification(vector<vector<double>> features, vector<string> ClassNames, vector<double> hu_Moments);

// Method that compares the features of the frame withe stored data using K Nearest Neighbour
string classifierKNN(vector<vector<double>> features, vector<string> ClassNames, vector<double> hu_Moments, int K);

// Method that reads the class id
string getClassid(char a);

// Method that writes the features into the CSV file
void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB);

// Method that loads the data from the CSV file to do prediction
void loadFromCSV(string filename, vector<string>& classNamesDB, vector<vector<double>>& featuresDB);


#endif#pragma once
#pragma once