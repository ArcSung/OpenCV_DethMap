#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
using namespace std;
using namespace cv::motempl;

class motdetect
{
private:
    // various tracking parameters (in seconds)
    const double MHI_DURATION = 10;
    const double MAX_TIME_DELTA = 1.0;
    const double MIN_TIME_DELTA = 0.5;
    // number of cyclic frame buffer used for motion detection
    // (should, probably, depend on FPS)

    // ring image buffer
    vector<Mat> buf;
    int last;

    // temporary images
    Mat mhi, orient, mask, segmask, zplane;
    vector<Rect> regions;

public:    
    motdetect();
    // parameters:
    //  img - input video frame
    //  dst - resultant motion picture
    //  args - optional parameters
    void  update_mhi(const Mat& img, Mat& dst, int diff_threshold);
};
