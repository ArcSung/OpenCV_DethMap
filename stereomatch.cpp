#include "stereomatch.hpp"

using namespace cv;
using namespace std;


static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match [--algorithm=bm|sgbm|hh] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}


void CalcuEDT(Mat DT, Mat Disp, Point ref)
{
    int channels = DT.channels();
    Mat EDT = Mat(DT.size(), DT.type(), Scalar(0));
    int refValue = DT.at<unsigned char>(ref.y, ref.x);
    if(refValue != 0)
    {    
        for(int y = 0; y < DT.rows - 1; y++)
            for(int x = 0; x < DT.cols*4 - 1; x++)
            {
                int Idp =  DT.at<unsigned char>(y, x);
                if(Idp != 0)
                {    
                    int diff = DT.at<unsigned char>(y, x)*(1 + abs(Idp - refValue)/refValue);
                    if(diff > 255)
                        diff = 255;
                    EDT.at<unsigned char>(y, x) = diff;
                }
            }   
        
        /*for(int y = 0; y < DT.rows; y++)
            for(int x = 0; x < DT.cols*4; x++)
            {
                EDT.at<unsigned char>(y, x) = DT.at<unsigned char>(y,x);
            }*/    
        imshow("EDT", EDT);
    }
}    

double CalcuDistance(Point P1, Point P2)
{
    return norm(P1 - P2);
}    

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
    int Thres = 95;
    if( !cascade.load(cascadeName)){ printf("--(!)Error cascade\n"); return -1; };
    if( !cascade2.load(cascadeName2)){ printf("--(!)Error cascade2\n"); return -1; };
    if( !cascade_hand.load(cascadeName3)){ printf("--(!)Error cascade3\n"); return -1; };

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

    namedWindow("disparity", 0);
    namedWindow("left", 0);        
    createTrackbar("Threshold", "disparity", &Thres, 256, 0);
    Mat bin_mask;

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
            bg_model1->apply(img1, fgmask1, update_bg_model ? -1 : 0);

            threshold(fgmask1, fgmask1, 0, 255, THRESH_BINARY + THRESH_OTSU);
            erode(fgmask1, fgmask1, Mat());
            dilate(fgmask1, fgmask1, Mat());
            fillContours(fgmask1);

            FrameCount++;
            if(FrameCount > 20)
                update_bg_model = false;
        }

        Mat disp, disp8, disp32F;

        if( alg == STEREO_BM )
            bm->compute(img1, img2, disp);
        else if( alg == STEREO_SGBM || alg == STEREO_HH )
            sgbm->compute(img1, img2, disp);

        if( alg != STEREO_VAR )
        {    
            disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
            disp.convertTo(disp32F, CV_32F, 1./16);
        }    
        else
            disp.convertTo(disp8, CV_8U);

        if(open_bg_model == true)
          disp8 = disp8 & fgmask1;
        inRange(disp8, Scalar(Thres, Thres, Thres), Scalar(256, 256, 256), fgmask1);
        disp8 = disp8 & fgmask1;
        flip(disp8, disp8, 1);

        flip(img2, img2, 1);
        flip(img1, img1, 1);

        detectAndDraw(img1, cascade, scale, disp8, bin_mask);

        imshow("left", img1);
        imshow("disparity", disp8);


        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c)
        {
            case 's':
                imwrite("bin_mask.png", bin_mask);
            break;    
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
    

    destroyAllWindows();

    camera0.release();
    camera1.release();

    return(0);
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, Mat disp, Mat& mask)
{
    int i = 0;
    char str[30];
    vector<Rect> faces;
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
    threshold(disp, mask, 10, 255, THRESH_BINARY);
    dilate(mask, mask, Mat());
    gray &= mask;
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
        if(r->width < img.cols/8)
            continue;
        BodySkeleton body_skeleton;
        Mat smallImgROI;
        Mat DT, Skin, dispMask = Mat::zeros(img.size(), CV_8UC1);
        Mat people = Mat::zeros(img.size(), CV_8UC3);
        Point center;
        Scalar color = colors[i%8];
        int radius = cvRound((r->width + r->height)*0.128*scale);
        double aspect_ratio = (double)r->width/r->height;

        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        double Dist = GetDistance(center.x, center.y, disp, mask, dispMask);
        threshold(disp, mask, 10, 255,  THRESH_BINARY);
        fillContours(mask);
        findConnectComponent(mask, center.x, center.y);
        img.copyTo(people, mask);
        Skin = findSkinColor(people);
        body_skeleton.head = Point(center.x, center.y);
        body_skeleton.neck = Point(center.x, center.y + r->height*0.6);
        body_skeleton.lShoulder = Point(0, 0);
        body_skeleton.rShoulder = Point(0, 0);
        findUpperBody( img, cascade2, scale, Rect(r->x, r->y, r->width, r->height), body_skeleton);
        if(body_skeleton.lShoulder.x != 0 && body_skeleton.lShoulder.y != 0 )
        {
            DT = findDistTran(mask);
            //find right arm
            //EDT = CalcuEDT(DT, body_skeleton.rShoulder);
            body_skeleton.rElbow = findArm(DT, body_skeleton.rShoulder, r->width*0.9, 0);
            body_skeleton.rHand = findHand(Skin, people, body_skeleton.rElbow, body_skeleton.head, r->height*1.5);

            //waitKey(0);
            //find left arm
            //EDT = CalcuEDT(DT, body_skeleton.lShoulder);
            body_skeleton.lElbow = findArm(DT, body_skeleton.lShoulder, r->width*0.9, 1);
            body_skeleton.lHand = findHand(Skin, people, body_skeleton.lElbow, body_skeleton.head, r->height*1.5);

            line(img, body_skeleton.head,   body_skeleton.neck, color, 2, 1, 0);
            line(img, body_skeleton.neck,   body_skeleton.rShoulder, color, 2, 1, 0);
            line(img, body_skeleton.neck,   body_skeleton.lShoulder, color, 2, 1, 0);
            line(img, body_skeleton.rShoulder,   body_skeleton.rElbow, color, 2, 1, 0);
            line(img, body_skeleton.lShoulder,   body_skeleton.lElbow, color, 2, 1, 0);
            line(img, body_skeleton.rElbow,   body_skeleton.rHand, color, 2, 1, 0);
            line(img, body_skeleton.lElbow,   body_skeleton.lHand, color, 2, 1, 0);
            circle(img, body_skeleton.head, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(img, body_skeleton.neck, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(img, body_skeleton.rShoulder, radius*0.2, Scalar(255, 0, 0), 2, 1, 0);
            circle(img, body_skeleton.lShoulder, radius*0.2, Scalar(255, 0, 0), 2, 1, 0);
            circle(img, body_skeleton.rElbow, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(img, body_skeleton.lElbow, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(img, body_skeleton.rHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
            circle(img, body_skeleton.lHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
        }    

        rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                    cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                    color, 3, 8, 0);
        if(Dist < 100)
            sprintf(str, "Dist: %2.2f cm",  Dist);
        else
        {
            Dist =  Dist/100;
            sprintf(str, "Dist: %2.2f m",  Dist);
        }  
        putText(img, str, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)), CV_FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 255, 0));
    }
}

double GetDistance(int x, int y, Mat disp8, Mat xyz, Mat &dispMask)
{
   double dispD = 0 ;
   double focal = disp8.cols*0.23;
   double between = 6.50; //the distance between 2 camera0
   int averge;

   for(int i = x - 2; i < x + 2; i++)
       for(int j = y - 2; j < y + 2; j++)
       {
           if(disp8.at<unsigned char>(y, x) != 0)
           {
               dispD += disp8.at<unsigned char>(y, x);
               averge++;
           }   
       }   

   if(dispD !=0)
   {
        dispD /= averge;
        inRange(disp8, Scalar(averge - 10), 255, dispMask);
        return (between*focal*16.0/dispD);
   } 
   else
    return 0;   
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
            fprintf(fp, "%f %f %f\n",  point[0], point[1], point[2]);
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
	int morph_size = 10; // 2*n + 1
    int operation = morph_operator + 2;

    // Apply the specified morphology operation
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( bw, bw, operation, element );
    
    vector<vector<Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
     
    findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
 
    Scalar color(255);
    for(int i = 0; i < contours.size(); i++) // Iterate through each contour
    {
            drawContours(bw, contours, i, color, CV_FILLED, 8, hierarchy);
    }
}

void findConnectComponent(Mat &bw, int x, int y)
{
    Mat labelImage(bw.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    int label = labelImage.at<int>(x, y);

    if(label > 0)
    {    
        inRange(labelImage, Scalar(label), Scalar(label), bw);
        threshold(bw, bw, 1, 255, THRESH_BINARY);
    }    

}

void findSkeleton(Mat bw)
{
    Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(9, 9));
    bool done;
    Mat skel(bw.size(), CV_8UC1, cv::Scalar(0));
    Mat temp(bw.size(), CV_8UC1);

    do
    {
        cv::morphologyEx(bw, temp, cv::MORPH_OPEN, element);
        cv::bitwise_not(temp, temp);
        cv::bitwise_and(bw, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        cv::erode(bw, bw, element);
 
        double max;
        cv::minMaxLoc(bw, 0, &max);
        done = (max == 0);
    } while (!done);

    //imshow("skeleton", skel);
}    

Mat findDistTran(Mat bw)
{
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1, NORM_MINMAX);
    dist.convertTo(dist, CV_8UC1, 255);
    //thinning(bw, dist);
    //namedWindow("Distance Transform Image", 0);
    //imshow("Distance Transform Image", dist);
    return dist;
}

void findUpperBody( Mat& img, CascadeClassifier& cascade,
                    double scale, Rect FaceRect,  BodySkeleton &body_skeleton)
{
    int i = 0;
    char str[30];
    vector<Rect> upbody;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, upbody,
        1.1, 2, 0
        |CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    for( vector<Rect>::const_iterator r = upbody.begin(); r != upbody.end(); r++, i++ )
    {
        if(r->width > FaceRect.width && r->height > FaceRect.height)
        {   
            Point center;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);

            if(center.x > FaceRect.x && center.x < FaceRect.x + FaceRect.width)
            {    
                printf("find upbody\n");
                body_skeleton.rShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.1), r->y*scale + (r->height-1)*0.9);
                body_skeleton.lShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.9), r->y*scale + (r->height-1)*0.9);
                /*rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                            cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                            Scalar(0, 255, 255), 3, 8, 0);*/
            }
        }    
    }
    printf("rShoulder: %d, %d   lShoulder: %d, %d\n", body_skeleton.rShoulder.x, 
            body_skeleton.rShoulder.y, body_skeleton.lShoulder.x, body_skeleton.lShoulder.y);
}

Mat findSkinColor(Mat src)
{
    Mat bgr2ycrcbImg, ycrcb2skinImg;
    cvtColor( src, bgr2ycrcbImg, cv::COLOR_BGR2HSV );
    inRange( bgr2ycrcbImg, cv::Scalar(0, 58, 40), cv::Scalar(35, 174, 255), ycrcb2skinImg );
    erode(ycrcb2skinImg, ycrcb2skinImg, Mat());
    dilate(ycrcb2skinImg, ycrcb2skinImg, Mat());
    //fillContours(ycrcb2skinImg);
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    findContours(ycrcb2skinImg,
    contours,
    hierarchy,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros( src.size(), CV_8UC1 );

    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>200)
        {
            drawContours(drawing, contours, i, Scalar(255), CV_FILLED, 8, hierarchy);
        }
    } 
    //imshow("skin color", drawing);

    return drawing;
}    

Point findArm(Mat EDT, Point lShoulder, int fheight, int findLeftelbow)
{
    float Slope = 0;
    float refValue = EDT.at<unsigned char>(lShoulder.x, lShoulder.y);
    Point elbow = lShoulder;
    Mat proc;
    GaussianBlur(EDT, proc, Size(5, 5), 0);
    inRange(proc, Scalar(refValue - 30 > 0? refValue - 30 : 2), Scalar(refValue + 3), proc);
    //threshold( proc, proc, 0, 255, THRESH_BINARY|THRESH_OTSU );
    //erode(proc, proc, Mat());
    //imshow("proc", proc);
    //return elbow;

    for(int i = 0; i < 5; i++)
    {
        bool find = false; 
        Point search;
        float TempSlope = 0;

        if(findLeftelbow == 1)
        {
            for(int x = elbow.x + fheight/4 > EDT.cols - 1 ? EDT.cols - 1 : elbow.x + fheight/4; x > (elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4); x--)
            //for(int x = elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4; x < (elbow.x + fheight/4 > EDT.cols-1 ? EDT.cols : elbow.x + fheight/4); x++)
            {
                for(int y = elbow.y + fheight/4 > EDT.rows - 1 ? EDT.rows - 1 : elbow.y + fheight/4; y > (elbow.y - fheight/4 < 0 ? 0 : elbow.y - fheight/4); y--)
                {
                  if(proc.at<unsigned char>(y, x) != 0
                    && x > lShoulder.x)
                  {
                       search = Point(x, y);
                       find = true;
                       break;
                  }    
                }

                if(find == true)
                {    
                  if(search.y - elbow.y !=0) 
                    TempSlope = (float)(search.x - elbow.x)/(search.y - elbow.y); 
                  else
                    TempSlope = search.x - elbow.x >= 0 ? 0.5 : -0.5;
                  break;
                }  
            }    
        }
        else
        {

            for(int x = elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4; x < (elbow.x + fheight/4 > EDT.cols-1 ? EDT.cols : elbow.x + fheight/4); x++)
            //for(int x = elbow.x + fheight/4 > EDT.cols - 1 ? EDT.cols - 1 : elbow.x + fheight/4; x > (elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4); x--)
            {
                for(int y = elbow.y + fheight/4 > EDT.rows - 1 ? EDT.rows - 1 : elbow.y + fheight/4; y > (elbow.y - fheight/4 < 0 ? 0 : elbow.y - fheight/4); y--)
                {
                  if(proc.at<unsigned char>(y, x) != 0
                    && x < lShoulder.x)
                  {
                       search = Point(x, y);
                       find = true;
                       break;
                  }    
                }

                if(find == true)
                {    
                  if(search.y - elbow.y !=0) 
                    TempSlope = (float)(search.x - elbow.x)/(search.y - elbow.y); 
                  else
                    TempSlope = search.x - elbow.x >= 0 ? 0.5 : -0.5;
                  break;
                }  
            }    
        }    

        printf("Slope %f, TempSlope %f\n", Slope, TempSlope);
        if(abs(Slope - TempSlope) > 0.4 && i > 3)
            break;     

        if(find == true)
        {
            Slope = TempSlope;
            elbow = search;
        }    
    }    

    printf("lelbow %d, %d\n", elbow.x, elbow.y);
    return elbow;
}    

Point findHand(Mat Skin, Mat People, Point rElbow, Point FacePoint, int FWidth)
{
    Point rHand = rElbow;
    Mat labelImage(Skin.size(), CV_32S);
    Mat mask(Skin.size(), CV_8UC1);
    Mat handimg(Skin.size(), CV_8UC3, Scalar(0));
    int nLabels = connectedComponents(Skin, labelImage, 8);
    int label;
    int minD = FWidth;
    int maxD = 0;
    int procD = 0;
    int facelabel = labelImage.at<int>(FacePoint.y, FacePoint.x);

    //find the most close area
    for(int x = rElbow.x - FWidth > 0 ? rElbow.x - FWidth: 0; x < (rElbow.x + FWidth < Skin.cols-1 ? rElbow.x + FWidth : Skin.cols -1); x++)
        for(int y = rElbow.y - FWidth > 0 ? rElbow.y - FWidth: 0; y < (rElbow.y + FWidth < Skin.rows-1 ? rElbow.y + FWidth : Skin.rows -1); y++)
        {
            if(labelImage.at<int>(y,x) != 0 && labelImage.at<int>(y,x) != facelabel)
            {    
                procD =CalcuDistance(rElbow, Point(x,y));
                if(procD < minD)
                {    
                    minD = procD;
                    label = labelImage.at<int>(y,x);
                }        
            }    
        } 

    inRange(labelImage, Scalar(label), Scalar(label), mask);
    Mat element_mask = Mat(Size(5, 5), CV_8UC1, Scalar(1));
    dilate(mask, mask, element_mask);
    People.copyTo(handimg, mask);
    //imshow("hand", handimg);
    double scale = 1.0;

    vector<Rect> hand;
    Mat gray, smallImg( cvRound (Skin.rows/scale), cvRound(Skin.cols/scale), CV_8UC1 );
    cvtColor( handimg, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    //imshow("smallImg", smallImg);

    cascade_hand.detectMultiScale( smallImg, hand,
        1.1, 2, 0
        |CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(10, 10) );

    for( vector<Rect>::const_iterator r = hand.begin(); r != hand.end(); r++)
    {
        rHand.x = cvRound((r->x + r->width*0.5)*scale);
        rHand.y = cvRound((r->y + r->height*0.5)*scale);
        return rHand;
        /*rectangle( People, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                    cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                    Scalar(0, 255, 255), 3, 8, 0);*/
    }
    //imshow("people", People);
    //find the most far point of the most close area
    FWidth = FWidth * 1.2;
    for(int x = rElbow.x - FWidth > 0 ? rElbow.x - FWidth: 0; x < (rElbow.x + FWidth < Skin.cols-1 ? rElbow.x + FWidth : Skin.cols -1); x++)
        for(int y = rElbow.y - FWidth > 0 ? rElbow.y - FWidth: 0; y < (rElbow.y + FWidth < Skin.rows-1 ? rElbow.y + FWidth : Skin.rows -1); y++)
        {
            if(labelImage.at<int>(y,x) == label)
            {    
                procD =CalcuDistance(rElbow, Point(x,y));
                if(procD > maxD)
                {    
                    maxD = procD;
                    rHand = Point(x,y);
                }        
            }    
        }    
  //printf("hand %d, %d\n", rHand.x, rHand.y);  

  return rHand;
}    
