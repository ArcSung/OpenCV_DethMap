#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

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

Ptr<StereoBM> bm = StereoBM::create(16,9);
Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
int SADWindowSize = 0, numberOfDisparities = 0;
enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
int alg = STEREO_SGBM;
string cascadeName = "1.xml";

static void saveXYZ(const char* filename, const Mat& mat);

bool read_file(const char* filename);

void init_parameter(Rect roi1, Rect roi2, Mat img);

void fillContours(Mat &bw);

double GetDistance(int x, int y, Mat disp8, Mat xyz);

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, Mat disp, Mat xyz);
