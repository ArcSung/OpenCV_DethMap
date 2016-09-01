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
#include "handGesture.hpp"
#include "Preprocess.hpp"

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
    int RFingerNum;
    int HeadWidth;
    int HeadHeight;
    double FaceDistance;
    HandGesture RHandGesture;
    HandGesture LHandGesture;
    Rect FaceRect;

    //function
    void init(Mat src,  Mat disp8, Mat &disp8Mask, Rect r, Rect RoiRect, double scale);
    void GetFaceDistance(Mat disp8, Mat &dispMask);
    void FindFaceConnect(Mat &bw);
    void FindUpperBody(CascadeClassifier& cascade, double scale);
    void FindArm(int RightOrLeft);
    void FindHand(Mat &img, CascadeClassifier& cascade_hand, int RightOrLeft);
    void ClearFingerNum(int RightOrLeft);

  private:  
    Mat PeopleSeg;
    Mat SkinSeg;
    Mat disp;
    Mat dispMask;

    double FaceDepth;
};

void findSkeleton(Mat &bw);



Mat CalcuEDT(Mat DT, Point ref);

Mat findSkinColor(Mat src);


Point findHand(Mat &img,  Mat Skin, Mat People, CascadeClassifier& cascade_hand, BodySkeleton &body_skeleton, int RightOrLeft);
