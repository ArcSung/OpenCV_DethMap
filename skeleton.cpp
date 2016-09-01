#include "skeleton.hpp"

using namespace cv;
using namespace std;

void BodySkeleton::init(Mat src, Mat disp8, Mat &disp8Mask, Rect r, Rect RoiRect, double scale)
{
    head = Point(cvRound((r.x + r.width*0.5)*scale - RoiRect.x), cvRound((r.y + r.height*0.5)*scale - RoiRect.y));
    neck = Point(head.x, head.y + r.height*0.6);
    HeadWidth = r.width;
    HeadHeight = r.height;
    lShoulder = Point(0, 0);
    rShoulder = Point(0, 0);
    FaceRect = r;

    disp8.copyTo(disp);
    disp8Mask.copyTo(dispMask);

    GetFaceDistance(disp, dispMask);
    FindFaceConnect(dispMask);
    
    PeopleSeg.setTo(Scalar(0, 0, 0));
    src.copyTo(PeopleSeg, dispMask);
    SkinSeg = findSkinColor(PeopleSeg);
}    

void BodySkeleton::GetFaceDistance(Mat disp8, Mat &dispMask)
{
   double dispD = 0;
   double focal = dispMask.cols*0.125;
   double between = 6.50; //the distance between 2 camera0
   int averge = 0;

   for(int i = head.x - 1; i <= head.x + 1; i++)
       for(int j = head.y - 1; j <= head.y + 1; j++)
       {
           if(disp8.at<unsigned char>(j, i) != 0)
           {
               dispD += disp8.at<unsigned char>(j, i);
               averge++;
           }   
       }   

   if(dispD !=0)
   {
        dispD /= averge;
        FaceDepth = dispD;
        inRange(disp8, Scalar(dispD - 5), Scalar(255), dispMask);
        fillContours(dispMask);
        FaceDistance = between*focal*16.0/dispD;
   } 
   else
        FaceDistance = 0;   
}

void BodySkeleton::FindFaceConnect(Mat &bw)
{
    Mat labelImage(bw.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    int label = labelImage.at<int>(head.x, head.y);

    if(label > 0)
    {    
        inRange(labelImage, Scalar(label), Scalar(label), bw);
        threshold(bw, bw, 0, 255, THRESH_BINARY);
    }    
}

void BodySkeleton::FindUpperBody(CascadeClassifier& cascade, double scale)
{
    int i = 0;
    char str[30];
    vector<Rect> upbody;
    Mat gray, smallImg( cvRound (PeopleSeg.rows/scale), cvRound(PeopleSeg.cols/scale), CV_8UC1 );
    cvtColor( PeopleSeg, gray, COLOR_BGR2GRAY );
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
        if(r->width > FaceRect.width && r->height > FaceRect.height && r->width < FaceRect.width*2.5 && r->height < FaceRect.height*2.5)
        {   
            Point center;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);

            if(center.x > FaceRect.x && center.x < FaceRect.x + FaceRect.width && center.y < FaceRect.y + FaceRect.width*2 )
            {    
                lShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.1), r->y*scale + (r->height-1)*0.9);
                rShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.9), r->y*scale + (r->height-1)*0.9);
                break;
            }
        }    
    }
}


void BodySkeleton::FindArm(int RightOrLeft)
{
    Mat EDT = FindDistTran(dispMask);
    int fheight = HeadHeight*0.9;
    float Slope = 0;
    Point elbow;
    if(RightOrLeft == 1)
        elbow = Point(rShoulder);
    else
        elbow = Point(lShoulder);

    Mat proc;
    GaussianBlur(EDT, proc, Size(5, 5), 0);

    for(int i = 0; i < 5; i++)
    {
        bool find = false; 
        Point search;
        float TempSlope = 0;

        if(RightOrLeft == 1)
        {
            for(int x = elbow.x + fheight/4 > EDT.cols - 1 ? EDT.cols - 1 : elbow.x + fheight/4; x > (elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4); x--)
            {
                for(int y = elbow.y + fheight/4 > EDT.rows - 1 ? EDT.rows - 1 : elbow.y + fheight/4; y > (elbow.y - fheight/4 < 0 ? 0 : elbow.y - fheight/4); y--)
                {
                  if(proc.at<unsigned char>(y, x) != 0
                    && x > rShoulder.x)
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

        if(abs(Slope - TempSlope) > 0.4 && i > 3)
            break;     

        if(find == true)
        {
            Slope = TempSlope;
            elbow = search;
        }    
    }    

    if(RightOrLeft)
        rElbow = elbow;
    else
        lElbow = elbow;
}    

void BodySkeleton::FindHand(Mat &img, CascadeClassifier& cascade_hand, int RightOrLeft)
{
    Point Hand;
    Point Elbow;
    Mat labelImage(SkinSeg.size(), CV_32S);
    Mat mask(SkinSeg.size(), CV_8UC1, Scalar(0));
    Mat handimg(SkinSeg.size(), CV_8UC3, Scalar(0));
    Mat Procimg(SkinSeg.size(), CV_8UC1, Scalar(0));
    HandGesture *_hg;
    int FWidth = HeadWidth;
    int label = 0;
    int minD = FWidth;
    int maxD = 0;
    int procD = 0;
    inRange(disp, Scalar(FaceDepth+5), Scalar(FaceDepth+10), Procimg);
    RemoveSmallRegion(Procimg, FWidth*FWidth/4);
    //imshow("Procimg", Procimg);
    int nLabels = connectedComponents(Procimg, labelImage, 8);
    vector<Point2f> ConnerPoint;
    if(RightOrLeft == 1)
    {   
        Hand  = Point(rElbow);
        Elbow = Point(rElbow);
        _hg   = &RHandGesture;
    }    
    else
    {
        Hand  = Point(lElbow);
        Elbow = Point(lElbow);
        _hg   = &LHandGesture;
    }    

    //find the most close area
    for(int x = Elbow.x - FWidth > 0 ? Elbow.x - FWidth: 0; x < (Elbow.x + FWidth < SkinSeg.cols-1 ? Elbow.x + FWidth : SkinSeg.cols -1); x++)
        for(int y = Elbow.y - FWidth > 0 ? Elbow.y - FWidth: 0; y < (Elbow.y + FWidth < SkinSeg.rows-1 ? Elbow.y + FWidth : SkinSeg.rows -1); y++)
        {
            if(labelImage.at<int>(y,x) != 0 )
            {    
                if(RightOrLeft == 0 && x > rShoulder.x)
                    continue;
                else if(RightOrLeft == 1 && x < lShoulder.x)
                    continue;

                procD = CalcuDistance(Elbow, Point(x,y));
                if(procD < minD)
                {    
                    minD = procD;
                    label = labelImage.at<int>(y,x);
                }        
            }    
        } 

    if(label != 0)
    {
        inRange(labelImage, Scalar(label), Scalar(label), mask);
        FWidth = FWidth * 2.5;
        for(int x = Elbow.x - FWidth > 0 ? Elbow.x - FWidth: 0; x < (Elbow.x + FWidth < SkinSeg.cols-1 ? Elbow.x + FWidth : SkinSeg.cols -1); x++)
            for(int y = Elbow.y - FWidth > 0 ? Elbow.y - FWidth: 0; y < (Elbow.y + FWidth < SkinSeg.rows-1 ? Elbow.y + FWidth : SkinSeg.rows -1); y++)
            {
                if(labelImage.at<int>(y,x) == label)
                //if(mask.at<int>(x,y) == 255)
                {    
                    procD = CalcuDistance(Elbow, Point(x,y));
                    if(procD > maxD)
                    {    
                        maxD = procD;
                        Hand = Point(x,y);
                    }        
                }    
            }    
        Mat CircleMask = Mat::zeros(mask.size(), CV_8UC1);
        circle(CircleMask, Hand, FWidth*0.4, Scalar(255), CV_FILLED, 1, 0);
        mask &= CircleMask;
        mask &= SkinSeg;
        //imshow("hand mask", mask);
        
        if(RightOrLeft == 1)
        {
            findContours(mask, _hg->contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
            _hg->frameNumber++;
            _hg->initVectors(); 
            _hg->cIdx=findBiggestContour(_hg->contours);
            if(_hg->cIdx!=-1){
                _hg->bRect=boundingRect(Mat(_hg->contours[_hg->cIdx]));		
                convexHull(Mat(_hg->contours[_hg->cIdx]),_hg->hullP[_hg->cIdx],false,true);
                convexHull(Mat(_hg->contours[_hg->cIdx]),_hg->hullI[_hg->cIdx],false,false);
                approxPolyDP( Mat(_hg->hullP[_hg->cIdx]),_hg->hullP[_hg->cIdx], 18, true );
                if(_hg->contours[_hg->cIdx].size()>3 ){
                    convexityDefects(_hg->contours[_hg->cIdx],_hg->hullI[_hg->cIdx],_hg->defects[_hg->cIdx]);
                    _hg->eleminateDefects(img, mask, HeadHeight);
                }
                bool isHand=_hg->detectIfHand();
                Moments mo = moments(_hg->contours[_hg->cIdx]);
                Hand = Point(mo.m10/mo.m00, mo.m01/mo.m00);
                if(Hand.y < rShoulder.y){	
                    RFingerNum = _hg->getFingerTips(img, mask, Hand, HeadHeight);
                    //_hg->drawFingerTips(img);
		        }
	        }
        }    
        else
        {
            findContours(mask, _hg->contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
            _hg->frameNumber++;
            _hg->initVectors(); 
            _hg->cIdx=findBiggestContour(_hg->contours);
            if(_hg->cIdx!=-1){
                //approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
                _hg->bRect=boundingRect(Mat(_hg->contours[_hg->cIdx]));		
                convexHull(Mat(_hg->contours[_hg->cIdx]),_hg->hullP[_hg->cIdx],false,true);
                convexHull(Mat(_hg->contours[_hg->cIdx]),_hg->hullI[_hg->cIdx],false,false);
                approxPolyDP( Mat(_hg->hullP[_hg->cIdx]),_hg->hullP[_hg->cIdx], 18, true );
                if(_hg->contours[_hg->cIdx].size()>3 ){
                    convexityDefects(_hg->contours[_hg->cIdx],_hg->hullI[_hg->cIdx],_hg->defects[_hg->cIdx]);
                    _hg->eleminateDefects(img, mask, HeadHeight);
                }
                bool isHand=_hg->detectIfHand();
                //hg->printGestureInfo(m->src);
                Moments mo = moments(_hg->contours[_hg->cIdx]);
                Hand = Point(mo.m10/mo.m00, mo.m01/mo.m00);
                if(Hand.y < lShoulder.y){	
                    _hg->getFingerTips(img, mask, Hand, HeadHeight);
                    _hg->getFingerNumber(img, mask);
                    //_hg->drawFingerTips(img);
		        }
	        }
        }    

    }    

    if(RightOrLeft)
    {
        if(Hand.x > lShoulder.x)
            rHand = Hand;
    }    
    else
        if(Hand.x < rShoulder.x)
            lHand = Hand;
}    

void BodySkeleton::ClearFingerNum(int RightOrLeft)
{
    HandGesture *_hg;
    if(RightOrLeft == 1)
        _hg   = &RHandGesture;
    else
        _hg   = &LHandGesture;

    _hg->Clear2DNumberDisplay();
}    
