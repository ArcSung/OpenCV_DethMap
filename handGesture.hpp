#ifndef _HAND_GESTURE_
#define _HAND_GESTURE_ 

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "Preprocess.hpp"

using namespace cv;
using namespace std;


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
		void eleminateDefects(Mat &src, Mat &bw, int FaceHeight);
		int  getFingerTips(Mat &src, Mat &bw, Point pHand, int FaceHeight);
		void drawFingerTips(Mat &src);
        void Clear2DNumberDisplay();
	private:
		string bool2string(bool tf);
		int fontFace;
		int prevNrFingerTips;
		void checkForOneFinger(Mat &src, Mat &bw);
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
		void removeRedundantEndPoints(vector<Vec4i> newDefects);
		void removeRedundantFingerTips();
};




#endif
