#ifndef _HAND_GESTURE_
#define _HAND_GESTURE_ 

#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

#define PI 3.14159

class HandGesture{
	public:
		HandGesture();
		vector<vector<Point> > contours;
		vector<vector<int> >hullI;
		vector<vector<Point> >hullP;
		vector<vector<Vec4i> > defects;	
		vector <Point> fingerTips;
		Rect rect;
		void printGestureInfo(Mat src);
		int cIdx;
		int frameNumber;
		int mostFrequentFingerNumber;
		int nrOfDefects;
		Rect bRect;
		double bRect_width;
		double bRect_height;
		bool isHand;
		bool detectIfHand();
		void initVectors();
		void getFingerNumber(Mat &src, Mat &bw);
		void eleminateDefects(Mat &src, Mat &bw);
		void getFingerTips(Mat &src, Mat &bw);
		void drawFingerTips(Mat &src);
	private:
		string bool2string(bool tf);
		int fontFace;
		int prevNrFingerTips;
		void checkForOneFinger(Mat &src, Mat &bw);
		float getAngle(Point s,Point f,Point e);	
		vector<int> fingerNumbers;
		void analyzeContours();
		string intToString(int number);
		void computeFingerNumber();
		void drawNewNumber(Mat &src, Mat &bw);
		void addNumberToImg(Mat &src);
		vector<int> numbers2Display;
		void addFingerNumberToVector();
		Scalar numberColor;
		int nrNoFinger;
		float distanceP2P(Point a,Point b);
		void removeRedundantEndPoints(vector<Vec4i> newDefects);
		void removeRedundantFingerTips();
};




#endif
