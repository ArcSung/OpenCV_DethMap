#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace std;

class BodySkeleton
{
  public:
    Point head;
    Point neck;
    Point lShoulder;
    Point rShoulder;
    Point rElbow;
    Point lElbow;
    Point rHand;
    Point lHand;

};


void findConnectComponent(Mat &bw, int x, int y);

void findSkeleton(Mat bw);

Mat findDistTran(Mat bw);

void findUpperBody( Mat& img, CascadeClassifier& cascade, double scale, Rect FaceRect, BodySkeleton &body_skeleton);

Mat CalcuEDT(Mat DT, Point ref);

Mat findSkinColor(Mat src);

Point findArm(Mat EDT, Point lShoulder, int fheight, int findLeftelbow);

Point findHand(Mat Skin, Mat People, CascadeClassifier& cascade_hand, Point rElbow, Point FacePoint, int FWidth);
