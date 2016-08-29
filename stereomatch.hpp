#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include "skeleton.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

/*camera parameter*/
Mat M1, D1, M2, D2;
Mat R, T, R1, P1, R2, P2;
Mat map11, map12, map21, map22;

CascadeClassifier cascade;
CascadeClassifier cascade2;
CascadeClassifier cascade_hand;

class _People{

public:
    int Faceindex;
    int Peopleindex;
    int LostFrame;
    int LostState;
    int ShoulderCount;
    int RHandCount;
    int LHandCount;
    //temp shoulder point
    Point lastRShoulder;
    Point lastLShoulder;
    Point lastRHand;
    Point lastLHand;
    Point locat;
    Rect FaceRect;
    BodySkeleton skeleton;
};

Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
int SADWindowSize = 0, numberOfDisparities = 0;
enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
int alg = STEREO_SGBM;
string cascadeName = "haar/1.xml";
string cascadeName2 = "haar/haarcascade_mcs_upperbody.xml";
string cascadeName3 = "haar/fist.xml";

const static Scalar colors[] =  { CV_RGB(0,0,255),
    CV_RGB(0,128,255),
    CV_RGB(0,255,255),
    CV_RGB(0,255,0),
    CV_RGB(255,128,0),
    CV_RGB(255,255,0),
    CV_RGB(255,0,0),
    CV_RGB(255,0,255)} ;

bool read_file(const char* filename);

void init_parameter(Rect roi1, Rect roi2, Mat img);

void fillContours(Mat &bw);

void FaceDetectAndTrack(Mat &img, DetectionBasedTracker &Detector,  vector<Rect> &Faces, vector<_People> &People);

double GetFaceDistance(int x, int y, Mat disp8, Mat &dispMask);

void detectAndDraw( Mat& img, Mat disp, double scale, vector<_People> &People);


