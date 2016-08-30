#include "stereomatch.hpp"
#include "motdetect.hpp"
#include "Preprocess.hpp"

using namespace cv;
using namespace std;

//motion detect
motdetect *_motdetect;


Rect bgmask;


int lastPeopleCount = 0;
Mat TrackMat;
Mat TrackMat2;
int PeopleCount = 0;


class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
    public:
        CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
        {
            CV_Assert(detector);
        }

        void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
        {
            Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
        }

        virtual ~CascadeDetectorAdapter()
        {}

    private:
        CascadeDetectorAdapter();
        cv::Ptr<cv::CascadeClassifier> Detector;
 };

static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match [--algorithm=bm|sgbm|hh] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
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
    int Thres = 85;
    if( !cascade.load(cascadeName)){ printf("--(!)Error cascade\n"); return -1; };
    if( !cascade2.load(cascadeName2)){ printf("--(!)Error cascade2\n"); return -1; };
    if( !cascade_hand.load(cascadeName3)){ printf("--(!)Error cascade3\n"); return -1; };

    _motdetect = new motdetect();

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
    Ptr<BackgroundSubtractor> bg_model1 =  
            createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
    Mat fgmask1, fgimg1;

    namedWindow("disparity", 0);
    namedWindow("left", 0);        
    createTrackbar("Threshold", "disparity", &Thres, 256, 0);
    Mat bin_mask;

    std::string cascadeFrontalfilename = "haarcascade_frontalface_alt.xml";
    cv::Ptr<cv::CascadeClassifier> cascadeTrack = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascadeTrack);

    cascadeTrack = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascadeTrack);

    DetectionBasedTracker::Parameters params;
    DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

    if (!Detector.run())
    {
        printf("Error: Detector initialization failed\n");
        return 2;
    }

    vector<Rect> Faces;
    vector<_People> People;
    TrackMat = Mat(img1.size(), CV_8UC1, Scalar(0));
    TrackMat2 = Mat(img1.size(), CV_8UC1, Scalar(0));

    while(1)
    {
        camera0 >> img1;
        camera1 >> img2;

        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;

        //cvtColor(img1, img1,CV_BGR2HSV);
        //cvtColor(img2, img2,CV_BGR2HSV);
        
        /*vector<Mat> channels1;
        vector<Mat> channels2;
        split(img1, channels1);
        split(img2, channels2);*/

        Mat disp, disp8, disp32F;

        //sgbm->compute(channels1[0], channels2[0], disp);
        sgbm->compute(img1, img2, disp);

        if( alg != STEREO_VAR )
        {    
            disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
            disp.convertTo(disp32F, CV_32F, 1./16);
        }    
        else
            disp.convertTo(disp8, CV_8U);

        /*fg bg segment*/
        if(open_bg_model)
        {    
            bg_model1->apply(img1, fgmask1, update_bg_model ? -1 : 0);
            //bg_model1->apply(img1, fgmask1, update_bg_model ? -1 : 0);

            threshold(fgmask1, fgmask1, 0, 255, THRESH_BINARY + THRESH_OTSU);
            erode(fgmask1, fgmask1, Mat());
            dilate(fgmask1, fgmask1, Mat());
            fillContours(fgmask1);

            FrameCount++;
            disp8 = disp8 & fgmask1;
            if(FrameCount > 20)
                update_bg_model = false;
        }

        //cvtColor(img1, img1,CV_HSV2BGR);
        //cvtColor(img2, img2,CV_HSV2BGR);
        Mat dispMask, dispTemp;
        inRange(disp8, Scalar(Thres, Thres, Thres), Scalar(256, 256, 256), dispMask);
        disp8.copyTo(dispTemp, dispMask);

        flip(dispTemp, disp8, 1);
        flip(img2, img2, 1);
        flip(img1, img1, 1);

        FaceDetectAndTrack(img1, Detector, Faces, People);
        detectAndDraw(img1, disp8, scale, People);
        //detectAndDraw(img1, cascade, scale, disp8, bin_mask);

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
        disp.release();
        disp8.release();
        disp32F.release();
        dispMask.release();
        dispTemp.release();
    }    
    

    destroyAllWindows();
    //_motdetect = ~motdetect();
    camera0.release();
    camera1.release();

    return(0);
}


void FaceDetectAndTrack(Mat &img,  DetectionBasedTracker &Detector,  vector<Rect> &Faces, vector<_People> &People)
{
    Mat GrayFrame, mask;
    cvtColor(img, GrayFrame, COLOR_RGB2GRAY);
    mask = findSkinColor(img);
    dilate(mask, mask, Mat());
    GrayFrame &= mask;
    equalizeHist( GrayFrame, GrayFrame);
    Detector.process(GrayFrame);
    Detector.getObjects(Faces);

        if(lastPeopleCount != Faces.size())
        {
            if(lastPeopleCount > Faces.size())
            {
                //printf("People.size() > Faces.size()\n");
                
                for(int s = 0; s< People.size();)
                {
                    if(TrackMat.at<unsigned char>(People[s].locat.y, People[s].locat.y) == 0)
                    {   
                        People[s].LostFrame++;
                        People[s].LostState = true;
                        //printf("LostFrame %d\n", People[s].LostFrame);
                        if(People[s].LostFrame > 3)
                        {    
                            People.erase(People.begin() + s);
                            continue;
                        }    
                    }    
                    ++s;
                }    
            }
            else if(lastPeopleCount < Faces.size()) //add People
            {
                //printf("People.size() < Faces.size()\n");
                int diff = Faces.size() - lastPeopleCount;
                int s = Faces.size();

                {
                    for(int i = 0; i < diff; i++)
                    {
                        if(TrackMat2.at<unsigned char>(Point(Faces[i+s].x + Faces[i+s].width*0.5, Faces[i+s].y + Faces[i+s].height*0.5)) != 0)
                        {

                            for(int w = 0; w< People.size();)
                            {
                                if(TrackMat2.at<unsigned char>(People[w].locat.y, People[w].locat.y) == People[w].Peopleindex)
                                {   
                                    People[s].LostFrame = 0;
                                    People[s].LostState = false;
                                    break;
                                }    
                            }    
                        }
                        else
                        {    
                            ++PeopleCount;
                            _People tempPeople;
                            int tempPindex =  PeopleCount;
                            int tempFindex =  i;
                            tempPeople.Peopleindex = tempPindex;
                            tempPeople.Faceindex = tempFindex;
                            tempPeople.locat = Point(Faces[i+s].x + Faces[i+s].width*0.5, Faces[i+s].y + Faces[i+s].height*0.5);
                            tempPeople.FaceRect = Faces[i+s];
                            tempPeople.LostFrame = 0;
                            tempPeople.LostState = false;
                            tempPeople.ShoulderCount = 10;
                            tempPeople.RHandCount = 10;
                            tempPeople.LHandCount = 10;
                            tempPeople.HoldMouse = false;
                            People.push_back(tempPeople);
                        }    
                    }    
                }    

            }    
        }    

        TrackMat.setTo(Scalar(0));
        TrackMat2.setTo(Scalar(0));
        int lostPeople = 0;
        //printf("before face for loop\n");
        for (size_t i = 0; i < Faces.size(); i++)
        {

            rectangle(TrackMat, Faces[i], Scalar(255), CV_FILLED, 8, 0);
            if(People[i + lostPeople].LostState == true)
            {
                lostPeople++;
            }else
            {    
                //update
                People[i + lostPeople].locat = Point(Faces[i].x + Faces[i].width*0.5, Faces[i].y + Faces[i].height*0.5);
                People[i + lostPeople].LostFrame = 0;
                People[i + lostPeople].LostState = false;
                People[i + lostPeople].FaceRect = Faces[i];
            }
        }
        //printf("after face for loop\n");

        lastPeopleCount = People.size();
        /*for(int s = 0; s< People.size(); s++)
        {
            rectangle(TrackMat2, People[s].FaceRect, Scalar(People[s].Peopleindex), CV_FILLED, 8, 0);
            if(People[s].LostState == false)
                lastPeopleCount++;
        } */   

        /*string text = format("People %d", lastPeopleCount);
        putText(img, text, Point(10, 20), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0,255,0), 2.0);
        text = format("PeopleCount %d", PeopleCount);
        putText(img, text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0,255,0), 2.0);*/
}

void detectAndDraw( Mat& img, Mat disp, double scale, vector<_People> &People)
{
    int i = 0;
    char str[30];

    for(int s = 0; s < People.size(); s++)
    {
        BodySkeleton *body_skeleton =  &People[s].skeleton;
        Rect r = People[s].FaceRect;
        Mat DT, EDT, Skin, dispMask, people, hand_mot = Mat(img.size(), CV_8UC1, Scalar(0));
        int pos_x = 0, pos_y = 0;
        //printf("r.x:%d, r.y:%d, r.width:%d, r.height:%d \n", r.x, r.y, r.width, r.height);

        //ROI setting
        Point tl, tr, bl, br;
        tl = Point (cvRound((r.x - r.width*2.0)*scale) > 0 ? cvRound((r.x - r.width*2.0)*scale) : 0
                , cvRound((r.y - r.height*2.0)*scale) > 0 ? cvRound((r.y - r.height*2.0)*scale) : 0); 
        tr = Point (cvRound((r.x + r.width*3.0)*scale) < img.cols ? cvRound((r.x + r.width*3.0)*scale):img.cols, tl.y); 
        bl = Point (tl.x, cvRound((r.y + r.height*5.0)*scale) < img.rows ? cvRound((r.y + r.height*5.0)*scale) : img.rows); 
        br = Point (tr.x, bl.y); 
        Rect RoiRect = Rect(tl.x, tl.y, tr.x - tl.x, bl.y - tl.y);
        Rect FaceRect = Rect(r.x, r.y, r.width, r.height);
        Mat imgROI = img(RoiRect);
        Mat handROI = hand_mot(RoiRect);
        Mat dispROI = disp(RoiRect);

        dispMask = Mat::zeros(imgROI.size(), CV_8UC1);
        people = Mat::zeros(imgROI.size(), CV_8UC3);

        // face Point
        Point center;
        Scalar color = colors[s%8];
        int radius = cvRound((r.width + r.height)*0.128*scale);
        double aspect_ratio = (double)r.width/r.height;
        center.x = cvRound((r.x + r.width*0.5)*scale - tl.x);
        center.y = cvRound((r.y + r.height*0.5)*scale - tl.y);

        body_skeleton->init(imgROI, dispROI, dispMask, FaceRect, RoiRect, scale);
        body_skeleton->FindUpperBody(cascade2, scale);

        if(People[s].ShoulderCount < 10 && body_skeleton->lShoulder.x == 0 && body_skeleton->lShoulder.y == 0 )
        {    
            body_skeleton->rShoulder = People[s].lastRShoulder;
            body_skeleton->lShoulder = People[s].lastLShoulder;
        }  
        else if(body_skeleton->lShoulder.x == 0 && body_skeleton->lShoulder.y == 0)
            People[s].ShoulderCount++;
            

        if(body_skeleton->lShoulder.x != 0 && body_skeleton->lShoulder.y != 0 )
        {
            People[s].lastRShoulder = body_skeleton->rShoulder;
            People[s].lastLShoulder = body_skeleton->lShoulder;
            People[s].ShoulderCount = 0;
            //findSkeleton(dispMask);
            //find right arm
            //EDT = CalcuEDT(DT, body_skeleton->rShoulder);
            body_skeleton->FindArm(1);
            body_skeleton->FindHand(imgROI, cascade_hand, 1);

            //waitKey(0);
            //find left arm
            //EDT = CalcuEDT(EDT, body_skeleton->lShoulder);
            body_skeleton->FindArm(0);
            body_skeleton->FindHand(imgROI, cascade_hand, 0);

#ifdef DeBug            
            line(imgROI, body_skeleton->head,   body_skeleton->neck, color, 2, 1, 0);
            line(imgROI, body_skeleton->neck,   body_skeleton->rShoulder, color, 2, 1, 0);
            line(imgROI, body_skeleton->neck,   body_skeleton->lShoulder, color, 2, 1, 0);
            line(imgROI, body_skeleton->rShoulder,   body_skeleton->rElbow, color, 2, 1, 0);
            line(imgROI, body_skeleton->lShoulder,   body_skeleton->lElbow, color, 2, 1, 0);
            circle(imgROI, body_skeleton->head, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(imgROI, body_skeleton->neck, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(imgROI, body_skeleton->rShoulder, radius*0.2, Scalar(255, 0, 0), 2, 1, 0);
            circle(imgROI, body_skeleton->lShoulder, radius*0.2, Scalar(255, 0, 0), 2, 1, 0);
            circle(imgROI, body_skeleton->rElbow, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
            circle(imgROI, body_skeleton->lElbow, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
#endif

            //right hand 
            if(CalcuDistance(body_skeleton->rElbow, body_skeleton->rHand) > r.width*0.5)
            {
                line(imgROI, body_skeleton->rElbow,   body_skeleton->rHand, color, 2, 1, 0);
                circle(imgROI, body_skeleton->rHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
                rectangle(hand_mot, cvPoint(body_skeleton->rHand.x - radius*0.5 + tl.x, body_skeleton->rHand.y - radius*0.5 + tl.y)
                        , cvPoint(body_skeleton->rHand.x + radius*0.5 + tl.x, body_skeleton->rHand.y + radius*0.5 + tl.y)
                        , Scalar(255), CV_FILLED, 8, 0);

                People[s].lastRHand = body_skeleton->rHand;
                People[s].RHandCount = 0;
                if(body_skeleton->RFingerNum >= 1)
                {
                    if(People[s].HoldMouse == false)
                    {
                        People[s].HoldMouse = true;
                        People[s].pMouseStart = body_skeleton->rHand;
                    }
                    else
                    {
                        People[s].pMouseEnd = body_skeleton->rHand;
                        int Distance = CalcuDistance(People[s].pMouseStart,  People[s].pMouseEnd); 
                        circle(imgROI, People[s].pMouseStart, Distance, Scalar(255, 255, 255), 2, 1, 0);
                        line(imgROI, People[s].pMouseStart,  People[s].pMouseEnd, Scalar(255, 255, 255), 2, 1, 0);
                    }
                }    
                else
                    People[s].HoldMouse = false;
            }
            else if(People[s].RHandCount < 3)
            {    
                body_skeleton->rHand = People[s].lastRHand;
                line(imgROI, body_skeleton->rElbow,   body_skeleton->rHand, color, 2, 1, 0);
                circle(imgROI, body_skeleton->rHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
                rectangle(hand_mot, cvPoint(body_skeleton->rHand.x - radius*0.5 + tl.x, body_skeleton->rHand.y - radius*0.5 + tl.y)
                    , cvPoint(body_skeleton->rHand.x + radius*0.5 + tl.x, body_skeleton->rHand.y + radius*0.5 + tl.y)
                    , Scalar(255), CV_FILLED, 8, 0);
                People[s].RHandCount++;

                if(body_skeleton->RFingerNum >= 1)
                {
                    if(People[s].HoldMouse == false)
                    {
                        People[s].HoldMouse = true;
                        People[s].pMouseStart = body_skeleton->rHand;
                    }
                    else
                    {
                        People[s].pMouseEnd = body_skeleton->rHand;
                        int Distance = CalcuDistance(People[s].pMouseStart,  People[s].pMouseEnd); 
                        circle(imgROI, People[s].pMouseStart, Distance, Scalar(255, 255, 255), 2, 1, 0);
                        line(imgROI, People[s].pMouseStart,  People[s].pMouseEnd, Scalar(255, 255, 255), 2, 1, 0);
                    }
                }    
                else
                    People[s].HoldMouse = false;
            } 
            else if(People[s].RHandCount > 3)
            { 
                People[s].HoldMouse = false;
                body_skeleton->ClearFingerNum(1);
            }    
            else
                People[s].RHandCount++;


            //left hand 
            if(CalcuDistance(body_skeleton->lElbow, body_skeleton->lHand) > r.width*0.5)
            {
                line(imgROI, body_skeleton->lElbow,   body_skeleton->lHand, color, 2, 1, 0);
                circle(imgROI, body_skeleton->lHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
                rectangle(hand_mot, cvPoint(body_skeleton->lHand.x - radius*0.5 + tl.x, body_skeleton->lHand.y - radius*0.5 + tl.y)
                        , cvPoint(body_skeleton->lHand.x + radius*0.5 + tl.x, body_skeleton->lHand.y + radius*0.5 + tl.y)
                        , Scalar(255), CV_FILLED, 8, 0);
                People[s].lastLHand = body_skeleton->lHand;
                People[s].LHandCount = 0;
            }
            else if(People[s].LHandCount < 3)
            {    
                body_skeleton->lHand = People[s].lastLHand;
                line(imgROI, body_skeleton->lElbow,   body_skeleton->lHand, color, 2, 1, 0);
                circle(imgROI, body_skeleton->lHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
                rectangle(hand_mot, cvPoint(body_skeleton->lHand.x - radius*0.5 + tl.x, body_skeleton->lHand.y - radius*0.5 + tl.y)
                    , cvPoint(body_skeleton->lHand.x + radius*0.5 + tl.x, body_skeleton->lHand.y + radius*0.5 + tl.y)
                    , Scalar(255), CV_FILLED, 8, 0);
                People[s].LHandCount++;
            }  
            else if(People[s].LHandCount > 3)
            {
                body_skeleton->ClearFingerNum(0);
            }    
            else
                People[s].LHandCount++;

            //_motdetect->update_mhi(img, hand_mot ,50);
        }    

        rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                    cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                    color, 3, 8, 0);
        sprintf(str, "%dP", People[s].Peopleindex);
        putText(img, str, cvPoint(cvRound(r.x*scale) + 10, cvRound(r.y*scale) + 25), CV_FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 255, 0));
        if(body_skeleton->FaceDistance < 100)
            sprintf(str, "D: %2.2f cm",  body_skeleton->FaceDistance);
        else
        {
            double Dist =  body_skeleton->FaceDistance/100;
            sprintf(str, "D: %2.2f m",  Dist);
        }  
        putText(img, str, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)), CV_FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 255, 0));
    }
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



