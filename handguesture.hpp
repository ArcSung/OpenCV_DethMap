#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
using namespace cv;

class Gesture
{
public:    
    void GestureDetection(Mat &fore, Mat &frame, Point &hand, int FaceHeight);

private:
    double dist(Point x,Point y);
    pair<Point,double> circleFromPoints(Point p1, Point p2, Point p3);
};    
