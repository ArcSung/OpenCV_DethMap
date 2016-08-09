#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"

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

static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match [--algorithm=bm|sgbm|hh] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

bool read_file(const char* filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", filename);
        return false;
    }

    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;
    fs["R"] >> R;
    fs["T"] >> T;

    return true;
}    

void init_parameter(Rect roi1, Rect roi2, Mat img)
{

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img.size().width/8) + 15) & -16;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(alg == STEREO_HH ? StereoSGBM::MODE_HH : StereoSGBM::MODE_SGBM);
}    

void fillContours(Mat &bw)
{
    // Another option is to use dilate/erode/dilate:
	int morph_operator = 1; // 0: opening, 1: closing, 2: gradient, 3: top hat, 4: black hat
	int morph_elem = 2; // 0: rect, 1: cross, 2: ellipse
	int morph_size = 3; // 2*n + 1
    int operation = morph_operator + 2;

    // Apply the specified morphology operation
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( bw, bw, operation, element );

}    

int GetDistance(int x, int y, Mat disp8)
{
    //get distance
   // printf("distance %d\n", disp8.at<unsigned char>(x, y));
   // cv::Mat_<float> vec_tmp(4,1);
   // vec_tmp(0)=x; vec_tmp(1)=y; vec_tmp(2)=dispa8.at<float>(x,y); vec_tmp(3)=1;
   // vec_tmp = Q*vec_tmp;
   // vec_tmp /= vec_tmp(3);
    //cv::Vec3f &point = XYZ.at<cv::Vec3f>(y,x);
    //point[0] = vec_tmp(0);
    //point[1] = vec_tmp(1);
    //point[2] = vec_tmp(2);
   // printf("vec_tmp(): %f, %f, %f \n", vec_tmp(0), vec_tmp(1), vec_tmp(2));
   return (disp8.at<unsigned char>(x,y));
}    

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, bool tryflip , Mat disp);

string cascadeName = "1.xml";

int main(int argc, char* argv[])
{

    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* nodisplay_opt = "--no-display";
    const char* scale_opt = "--scale=";

    const char* intrinsic_filename = 0;
    const char* point_cloud_filename = 0;

    bool no_display = false;
    float scale = 1.f;
    bool update_bg_model = true;
    bool open_bg_model = false;
    int FrameCount = 0;
    int Thres = 128;
    CascadeClassifier cascade;
    if( !cascade.load(cascadeName)){ printf("--(!)Error cascade\n"); return -1; };

    for( int i = 1; i < argc; i++ )
    {
        if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
                  strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  strcmp(_alg, "hh") == 0 ? STEREO_HH :
                  strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                return -1;
            }
        }
        else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
                numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
            {
                printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
                return -1;
            }
        }
        else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
                SADWindowSize < 1 || SADWindowSize % 2 != 1 )
            {
                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
        }
        else if( strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(scale_opt), "%f", &scale ) != 1 || scale < 0 )
            {
                printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
                return -1;
            }
        }
        else if( strcmp(argv[i], nodisplay_opt) == 0 )
            no_display = true;
        else if( strcmp(argv[i], "-i" ) == 0 )
            intrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-p" ) == 0 )
            point_cloud_filename = argv[++i];
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }


    Mat img1, img2, gray1, gray2;
    cv::VideoCapture camera0(0);
    cv::VideoCapture camera1(1);

    if( !camera0.isOpened() ) return 1;
    if( !camera1.isOpened() ) return 1;

    camera0 >> img1;
    camera1 >> img2;


    int color_mode = alg == STEREO_BM ? 0 : -1;
    if(color_mode == 0)
    {
        cvtColor(img1, img1, CV_RGB2GRAY);
        cvtColor(img2, img2, CV_RGB2GRAY);
    }    

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;


    init_parameter(roi1, roi2, img1);

    if(!read_file(intrinsic_filename))
        return -1;


    M1 *= scale;
    M2 *= scale;


    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    Mat img1r, img2r;

    //fg bg segment
    Ptr<BackgroundSubtractor> bg_model2 =  
            createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

    Ptr<BackgroundSubtractor> bg_model1 =  
            createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
    Mat fgmask1, fgimg1;
    std::vector<Mat> vectorOfHSVImages;

    namedWindow("disparity", 0);
    createTrackbar("Threshold", "disparity", &Thres, 256, 0);
    Mat xyz;

    while(1)
    {
        camera0 >> img1;
        camera1 >> img2;

        if(color_mode == 0)
        {
           cvtColor(img1, img1, CV_RGB2GRAY);
           cvtColor(img2, img2, CV_RGB2GRAY);
        }

        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;

        /*fg bg segment*/
        if(open_bg_model)
        {    
            if(FrameCount % 3 == 0)
                bg_model1->apply(img1, fgmask1, -1);
            else
                bg_model1->apply(img1, fgmask1, -1);

            threshold(fgmask1, fgmask1, 0, 255, THRESH_BINARY + THRESH_OTSU);
            erode(fgmask1, fgmask1, Mat());
            dilate(fgmask1, fgmask1, Mat());
            fillContours(fgmask1);

            FrameCount++;
            if(FrameCount > 1000)
                FrameCount = 0;
        }

        Mat disp, disp8, disp32F;
        //Mat img1p, img2p, dispp;
        //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
        //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

        int64 t = getTickCount();
        if( alg == STEREO_BM )
            bm->compute(img1, img2, disp);
        else if( alg == STEREO_SGBM || alg == STEREO_HH )
            sgbm->compute(img1, img2, disp);
        t = getTickCount() - t;
        //printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

        //disp = dispp.colRange(numberOfDisparities, img1p.cols);
        if( alg != STEREO_VAR )
        {    
            disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
            disp.convertTo(disp32F, CV_32F, 1./16);
        }    
        else
            disp.convertTo(disp8, CV_8U);

        reprojectImageTo3D(disp, xyz, Q, true);
        if(open_bg_model == true)
          disp8 = disp8 & fgmask1;
        inRange(disp8, Scalar(Thres, Thres, Thres), Scalar(256, 256, 256), fgmask1);
        disp8 = disp8 & fgmask1;
        //dilate(disp8, disp8, Mat());
        flip(disp8, disp8, 1);

        flip(img1, img1, 1);

        detectAndDraw(img1, cascade, 1, false, disp8);

        namedWindow("left", 1);        
        imshow("left", img1);
        namedWindow("right", 1);
        imshow("right", img2);
        imshow("disparity", disp8);
        //namedWindow("xyz", 0);
        //imshow("xyz", xyz);


        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c)
        {
            case 'c':
                update_bg_model = true;
                FrameCount = 0;
            break;

            case 'b':
                if(open_bg_model == true)
                    open_bg_model = false;
                else
                {    
                    open_bg_model = true;
                    update_bg_model = true;
                    FrameCount = 0;
                }    
            default:
            ;
        }
    }    
    
    saveXYZ("cloudfile", xyz);

    destroyAllWindows();

    camera0.release();
    camera1.release();
    return(0);
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, bool tryflip , Mat disp)
{
    int i = 0;
    char str[30];
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        double aspect_ratio = (double)r->width/r->height;

        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
        int dist = GetDistance(center.x, center.y, disp);
        sprintf(str, "dist: %d", dist);
        putText(img, str, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)), CV_FONT_HERSHEY_DUPLEX, 2, color);
    }
}
