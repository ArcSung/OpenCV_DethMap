// Preprocess.h

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
#define PI 3.14159

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
const int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
const int ADAPTIVE_THRESH_WEIGHT = 9;

// function prototypes ////////////////////////////////////////////////////////////////////////////

void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh);

cv::Mat extractValue(cv::Mat &imgOriginal);

cv::Mat maximizeContrast(cv::Mat &imgGrayscale);

cv::Mat CalcuEDT(cv::Mat DT, cv::Point ref);

void fillContours(cv::Mat &bw);

double CalcuDistance(Point P1, Point P2);

Mat FindDistTran(Mat bw);

int findBiggestContour(vector<vector<Point> > contours);

float getAngle(Point s, Point f, Point e);

#endif	// PREPROCESS_H

