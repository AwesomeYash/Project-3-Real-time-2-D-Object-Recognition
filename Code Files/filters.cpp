/*
Project 3: Object Detector
Author: Priyanshu Ranka
NUID: 002305396
Date: 02/19/2025
Description: This file contains the image processing functions.
*/

// Include Directories
#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>

#include "filters.h"

// Namespace
using namespace cv;
using namespace std;

// Function to save frame
string saveFrame(cv::Mat& frame, std::string path, std::string filename)
{
    std::string fileName = path + "/" + filename + ".jpg";
    cv::imwrite(fileName, frame);
    return fileName;
}

// Task 1: Thresholding function
Mat thresholdImage(Mat &image)
{   
	int threshold = 110;	// Considering threshold about half the total intensity
	Mat grayFrame, outputImage, frame;
	outputImage = Mat(image.size(), CV_8UC1);

	// Convert to Grayscale and Blur the image
    cvtColor(image, grayFrame, cv::COLOR_BGR2GRAY);
    GaussianBlur(outputImage, outputImage, Size(3, 3), 0);

	//In this loop the parts with intensity greater than threshold are set to 255 (white) and the rest are set to 0 (black)
    for (int i = 0; i < grayFrame.rows; i++)
	{
		for (int j = 0; j < grayFrame.cols; j++)
		{
			if (grayFrame.at<uchar>(i, j) <= threshold)
				outputImage.at<uchar>(i, j) = 255;

			else
				outputImage.at<uchar>(i, j) = 0;
		}
	}
    
	return outputImage;     // The outputImage is thresholded image
}

// Task 2: Cleaning the image
Mat cleanImage(Mat &image)
{
    Mat outputImage = Mat(image.size(), CV_8UC1);
	outputImage = thresholdImage(image); // Threshold the image

    // Apply morphological operations- Dilationa and Erosion
    Mat dilatedImage = dilateImage(outputImage, Size(2, 2), Point(-1, -1), 2);
    Mat erodedImage = erodeImage(dilatedImage, Size(2.5, 2.5), Point(-1, -1), 2);
    
	return erodedImage;	 //The output will be erode(dilate(image))

	// Alternative method to clean image
    /*
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(outputImage, frame, MORPH_OPEN, kernel); // Removes noise
    
	return frame;
     */
}

// Dilation of image function
Mat dilateImage(Mat inputImage, Size kernelSize, Point anchor, int iterations)
{
    // Create structuring element for dilation
    Mat kernel = getStructuringElement(MORPH_CROSS, kernelSize, anchor);

    // Perform dilation
    Mat outputImage;
    dilate(inputImage, outputImage, kernel, anchor, iterations);

    return outputImage;
}

// Erosion of image function
Mat erodeImage(Mat inputImage, Size kernelSize, Point anchor, int iterations)
{
    // Create structuring element for erosion
    Mat kernel = getStructuringElement(MORPH_CROSS, kernelSize, anchor);

    // Perform erosion
    Mat outputImage;
    erode(inputImage, outputImage, kernel, anchor, iterations);

    return outputImage;
}

// Task 3 and 4: Region Segmentation
// Function to generate random colors
vector<cv::Vec3b> generateRandomColors(int num_colors) {
    std::vector<cv::Vec3b> colors;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < num_colors; ++i) {
        colors.push_back(cv::Vec3b(dist(rng), dist(rng), dist(rng)));
    }
    return colors;
}

// Connected Components Analysis
Mat connectedComponentsAnalysis(const cv::Mat& thresholded_image)
{
    cv::Mat labels, stats, centroids;
    int min_size = 5000, max_regions = 3;
    // Compute connected components with statistics
    int num_labels = cv::connectedComponentsWithStats(thresholded_image, labels, stats, centroids, 8, CV_32S);

    // Store valid component indices
    std::vector<int> valid_indices;
    for (int i = 1; i < num_labels; ++i)
    { // Ignore background (0)
        if (stats.at<int>(i, cv::CC_STAT_AREA) > min_size)
        {
            valid_indices.push_back(i);
        }
    }

    // Sort components by area (largest first) and limit to max_regions
    std::sort(valid_indices.begin(), valid_indices.end(), [&](int a, int b)
        {
            return stats.at<int>(a, cv::CC_STAT_AREA) > stats.at<int>(b, cv::CC_STAT_AREA);
        });

    if (valid_indices.size() > max_regions)
    {
        valid_indices.resize(max_regions);
    }

    // Generate random colors
    std::vector<cv::Vec3b> colors = generateRandomColors(valid_indices.size());

    // Ensure output is a 3-channel image initialized to black
    cv::Mat output(thresholded_image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // Assign colors to valid components
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        int region_id = valid_indices[i];
        cv::Vec3b color = colors[i]; // Get the color for this region

        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                if (labels.at<int>(y, x) == region_id) {
                    output.at<cv::Vec3b>(y, x) = color; // Assign color
                }
            }
        }
    }

    return output;
}

//Method that identifies the regions of the image
Mat regions(Mat& image, Mat& regionLabels, Mat& stats, Mat& centroids, vector<int>& topNLabels) {
    Mat processedImage;
    int componentLabels = connectedComponentsWithStats(image, regionLabels, stats, centroids);

    Mat areas = Mat::zeros(1, componentLabels - 1, CV_32S);
    Mat sortedIdx;
    for (int i = 1; i < componentLabels; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        areas.at<int>(i - 1) = area;
    }
    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
    }

    vector<Vec3b> colors(componentLabels, Vec3b(0, 0, 0));

    int N = 1;
    if (N < sortedIdx.cols)
    {
        N = N;
    }
    else
    {
        N = sortedIdx.cols;
    }
    int THRESHOLD = 5000;
    for (int i = 0; i < N; i++)
    {
        int label = sortedIdx.at<int>(i) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > THRESHOLD)
        {
            colors[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            topNLabels.push_back(label);
        }
    }

    processedImage = Mat::zeros(regionLabels.size(), CV_8UC3);
    for (int i = 0; i < processedImage.rows; i++)
    {
        for (int j = 0; j < processedImage.cols; j++)
        {
            int label = regionLabels.at<int>(i, j);
            processedImage.at<Vec3b>(i, j) = colors[label];
        }
    }
    return processedImage;
}

// Method that calculates the bounding box
RotatedRect obtainBoundingBox(Mat& region, double x, double y, double alpha) {
    int maxX = INT_MIN;
    int minX = INT_MAX;
    int maxY = INT_MIN;
    int minY = INT_MAX;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int x1 = i * cos(alpha) - x * cos(alpha) + j * sin(alpha) - y * sin(alpha);
                int y1 = -i * sin(alpha) + x * sin(alpha) + j * cos(alpha) - y * cos(alpha);
                maxX = max(maxX, x1);
                minX = min(minX, x1);
                maxY = max(maxY, y1);
                minY = min(minY, y1);
            }
        }
    }
    int lx = maxX - minX;
    int ly = maxY - minY;
    if (lx < ly) {
        int temp = lx;
        lx = ly;
        ly = temp;
    }

    Point centroid = Point(x, y);
    Size size = Size(lx, ly);

    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}

// Method that draws the axis line
void sketchLine(Mat& image, double x, double y, double theta, Scalar color) {
    double l = 100.0;
    double xdot = x + sqrt(l * l - (l * sin(theta)) * (l * sin(theta)));
    double ydot = y + l * sin(theta);

    arrowedLine(image, Point(x, y), Point(xdot, ydot), color, 3);
}

// Method that draws the bounding box
void sketchBoundingBox(Mat& image, RotatedRect boundingBox, Scalar color) {
    Point2f rect_points[4];
    boundingBox.points(rect_points);
    for (int i = 0; i < 4; i++) {
        Point start = rect_points[i];
        Point end = rect_points[(i + 1) % 4];
        // Draw the line on the image
        line(image, start, end, color, 3);
    }
}

// Task 5 onwards: Classification
// Method that calculates the features
void calculateHu_Moments(Moments mo, vector<double>& hu_Moments) {
    double huMo[7];
    //double d;
    HuMoments(mo, huMo);
    for (double d : huMo) {
        hu_Moments.push_back(d);
    }
    return;
}

// Method that calculates the Euclidean Distance
double computeEuclideanDistance(vector<double> feats1, vector<double> feats2) {
    double sumfeats1 = 0;
    double sumfeats2 = 0;
    double sumfeatsDifference = 0;
    for (int i = 0; i < feats1.size(); i++) {
        double diff = feats1[i] - feats2[i];
        // Square the difference and add it to the sum of squared differences
        double squaredDiff = diff * diff;
        sumfeatsDifference += squaredDiff;
        sumfeats1 = sumfeats1 + feats1[i] * feats1[i];
        sumfeats2 = sumfeats2 + feats2[i] * feats2[i];
    }
    double eucliduan = sqrt(sumfeatsDifference) / (sqrt(sumfeats1) + sqrt(sumfeats2));
    return eucliduan;
}

// Method that compares the features of the frame withe stored data
string classification(vector<vector<double>> features, vector<string> ClassNames, vector<double> hu_Moments) 
{
    double THRESHOLD = 0.2;
    double dist = DBL_MAX;
    string name = " ";
    for (int i = 0; i < features.size(); i++) 
    {
        vector<double> dbFeature = features[i];
        string storedClassName = ClassNames[i];
        double eucDistance = computeEuclideanDistance(dbFeature, hu_Moments);
        if (eucDistance < dist && eucDistance < THRESHOLD) 
        {
            name = storedClassName;
            dist = eucDistance;
        }
    }
    return name;
}

// Method that compares the features of the frame withe stored data using K Nearest Neighbour
string classifierKNN(vector<vector<double>> features, vector<string> ClassNames, vector<double> hu_Moments, int K) 
{
    double THRESHOLD = 0.15;
    vector<double> dist;
    for (int i = 0; i < features.size(); i++) 
    {
        vector<double> eachFeature = features[i];
        double euc_dist = computeEuclideanDistance(eachFeature, hu_Moments);
        if (euc_dist < THRESHOLD) 
        {
            dist.push_back(euc_dist);
        }
    }

    string className = " ";
    if (dist.size() > 0) 
    {
        int n = dist.size();
        vector<int> sortedIdx(n);
        for (int i = 0; i < n; i++) 
        {
            sortedIdx[i] = i;
        }

        for (int i = 0; i < n - 1; i++) 
        {
            for (int j = 0; j < n - i - 1; j++) 
            {
                if (dist[sortedIdx[j]] > dist[sortedIdx[j + 1]])
                {
                    swap(sortedIdx[j], sortedIdx[j + 1]);
                }
            }
        }

        vector<string> firstKNames;
        int s = sortedIdx.size();
        vector<int> nameCount(ClassNames.size(), 0);
        int range = min(s, K);
        for (int i = 0; i < range; i++) 
        {
            string name = ClassNames[sortedIdx[i]];
            int nameIdx = find(ClassNames.begin(), ClassNames.end(), name) - ClassNames.begin();
            nameCount[nameIdx]++;
        }

        int maxCount = 0;
        for (int i = 0; i < nameCount.size(); i++) 
        {
            if (nameCount[i] > maxCount) {
                className = ClassNames[i];
                maxCount = nameCount[i];
            }
        }
    }
    return className;
}

//Method that takes the class name and id
string getClassid(char a) 
{
    if (a == 'a')
    {
		return "mug";
	}
    else if (a == 'b')
    {
        return "bottle";
    }
	else if (a == 'c')
	{
		return "butterfly";
	}
	else if (a == 't')
	{
		return "thermos";
	}
	else if (a == 'e')
	{
		return "earphones";
	}
	else if (a == 'f')
	{
		return "football";
	}
	else if (a == 'h')
	{
		return "headphones";
	}
	else if (a == 'k')
	{
		return "keys";
	}
	else if (a == 'j')
	{
		return "tiffin box";
	}
	else if (a == 'p')
	{
		return "pen";
	}
	else
	{
		return "Nothing found";
	}
}

// CSV functions
// Method that writes the features into the CSV file
void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB) 
{
    ofstream csvFile;
    csvFile.open(filename, ofstream::trunc);

    for (int i = 0; i < classNamesDB.size(); i++) 
    {
        csvFile << classNamesDB[i] << ",";
        for (int j = 0; j < featuresDB[i].size(); j++) 
        {
            csvFile << featuresDB[i][j];
            if (j != featuresDB[i].size() - 1) 
            {
                csvFile << ",";
            }
        }
        csvFile << "\n";
    }
}

// Method that loads the data from the CSV file to do prediction
void loadFromCSV(string filename, vector<string>& classNamesDB, vector<vector<double>>& featuresDB) 
{
    ifstream csvFile(filename);
    if (csvFile.is_open()) 
    {
        string line;
        while (getline(csvFile, line)) 
        {
            vector<string> currLine;
            int pos = 0;
            string token;
            while ((pos = line.find(",")) != string::npos) 
            {
                token = line.substr(0, pos);
                currLine.push_back(token);
                line.erase(0, pos + 1);
            }
            currLine.push_back(line);

            vector<double> currFeature;
            if (currLine.size() != 0) 
            {
                classNamesDB.push_back(currLine[0]);
                for (int i = 1; i < currLine.size(); i++) 
                {
                    currFeature.push_back(stod(currLine[i]));
                }
                featuresDB.push_back(currFeature);
            }
        }
    }
}
