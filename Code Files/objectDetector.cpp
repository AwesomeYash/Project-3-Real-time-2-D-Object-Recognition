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

    cout << "Enter the type of Classification: (n) Normal Classification OR (k) KNN with K = 3" << endl;
    const char keyPress = waitKey(10);     // Key press for training between KNN and classification

    // Grabbing the frame
    std::cout << "Start grabbing" << endl;

    // Keypress and frame number
    int frameNumber = 0;
	int lastKeyPressed = 0;	            // Last key pressed to save and quit
	
	struct RegionFeatures rf;

    // Loop to capture frames
    for (;;)
    {
        // Taking the basic video capture from Project_1
        cap.read(frame);
        
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        Mat filteredFrame = frame.clone();
        
		// Task 1: Thresholding Image
        //filteredFrame = thresholdImage(filteredFrame);
        
        // Task 2: Cleanning Image
        filteredFrame = cleanImage(filteredFrame);

		// Task 3: Region Segmentation
        Mat region_map = connectedComponentsAnalysis(filteredFrame);

		// Task 4: Region Features
        Mat Regions, stats, centroids;
        vector<int> CountTopLabels;

        Mat regionFrame = regions(filteredFrame, Regions, stats, centroids, CountTopLabels);

        // Task 5 and 6: Classifiying new images and giving labels
        vector<string> ClassName;
        vector<vector<double>> features;
        Mat img;
        bool training = false;

	    // Input for switching to training
        const char lastKeyPressed2 = waitKey(10);     // Key press for training between KNN and classification
        training = (lastKeyPressed2 == 't') ? !training : training;
        cout << (training ? "Training" : "Testing") << endl;

        // loads data from a CSV file
        loadFromCSV("./features.csv", ClassName, features);

		// Task 5, 6, 7: Classifying new images and giving labels
        for (int n = 0; n < CountTopLabels.size(); n++)
        {
            int label = CountTopLabels[n];
            Mat region;
            region = (Regions == label);

            Moments m = moments(region, true);
            double CX = centroids.at<double>(label, 0);
            double CY = centroids.at<double>(label, 1);
            double theta = atan2(m.mu11, 0.5 * (m.mu20 - m.mu02));
            
            //function call to draw bounding box
            RotatedRect BoundingBox = obtainBoundingBox(region, CX, CY, theta);
            sketchLine(frame, CX, CY, theta, Scalar(0, 0, 255));
            sketchBoundingBox(frame, BoundingBox, Scalar(0, 255, 0));
            
            // function call to calculate the features
            vector<double> hu_Moments;
            calculateHu_Moments(m, hu_Moments); 
            
			// Training mode
            if (training)
            {
                namedWindow("Current Region", WINDOW_AUTOSIZE);
                imshow("Current Region", region);

                cout << "Enter the class that the image represents" << endl;
                char k = waitKey(0);
                string className = getClassid(k);

                features.push_back(hu_Moments);
                ClassName.push_back(className);

                if (n == CountTopLabels.size() - 1)
                {
                    training = false;
                    cout << "Testing Mode" << endl;
                    destroyWindow("Current Region");
                }
            }
            
            else 
            {
                // condition to do prediction of the model
                string className;
                if (keyPress == 'n')
                {   // calling the classification methos
					cout << "Classification" << endl;
                    className = classification(features, ClassName, hu_Moments);
					//waitKey(0);
                }
                else if (keyPress == 'k')
                {   // calling the KNearest Neighbour
                    className = classifierKNN(features, ClassName, hu_Moments, 3);
					cout << "KNN" << endl;
					//waitKey(0);
                }
            }
            putText(frame, "headphones", Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        }

        // Displaying Outputs
		// Display the original frame 
        cv::namedWindow("Original frame");
        cv::imshow("Original frame", frame);
		// Display the filtered frame
        cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
        cv::imshow("Region Map", regionFrame);
        // Display region features
        cv::namedWindow("Region Features", cv::WINDOW_AUTOSIZE);
        cv::imshow("Region Features", filteredFrame);

        // Create filename for saving the frame and region map
        std::string filename = "frame" + std::to_string(frameNumber);
        std::string regionMap = "region" + std::to_string(frameNumber);
        std::string regionFeature = "features" + std::to_string(frameNumber);

        int keyPress = cv::waitKey(10);
        // quit, save, or set filter
        switch (keyPress)
        {
        case 'q':
            writeToCSV("./features.csv", ClassName, features);
            cap.release();
            destroyAllWindows();
            return 0;
			
        case 's':
            // increment frame number
            saveFrame(frame, ".", filename);
            saveFrame(region_map, ".", regionFeature);
            saveFrame(filteredFrame, ".", regionMap);
            break;

        default:
            // Sets last valid key pressed for filter selection
            lastKeyPressed = (keyPress != -1) ? keyPress : lastKeyPressed;
            break;
        }
        frameNumber++;
    }
    
    // [From OpenCV] the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}