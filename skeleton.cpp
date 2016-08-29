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
    
    src.copyTo(PeopleSeg, dispMask);
    SkinSeg = findSkinColor(PeopleSeg);
}    

void BodySkeleton::GetFaceDistance(Mat disp8, Mat &dispMask)
{
   double dispD = 0;
   double focal = 480*0.23;
   double between = 6.50; //the distance between 2 camera0
   int averge;

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
        inRange(disp8, Scalar(dispD - 15), Scalar(255), dispMask);
        fillContours(dispMask);
        FaceDistance = between*focal*16.0/dispD;
   } 
   else
        FaceDistance = 0;   
}

bool FindHandCorner(Mat bin_img, std::vector<Point2f> &ConnerPoint)
{
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    Point2f L1, L2;

    findContours(bin_img,
    contours,
    hierarchy,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE);

    int MaxSize = 0, MaxSizeId = 0;

    if(contours.size() == 0)
        return false;

    //Find the max contour
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>MaxSize && a < (bin_img.cols*bin_img.rows)/4)
        {
            MaxSize=a;
            MaxSizeId=i;                //Store the index of largest contour
        }
    } 

    if(MaxSize < 100)
        return false;

    vector< vector< Point> > contours_poly(contours.size());
    approxPolyDP( Mat(contours[MaxSizeId]), contours_poly[MaxSizeId], 3, true ); // let contours more smooth
  
    //find 4 conner in contour
    int tlx = bin_img.cols, tly = bin_img.rows, brx = 0, bry = 0;

    for(int j=0;j<contours_poly[MaxSizeId].size();j++)
    {
        if(tlx < contours_poly[MaxSizeId][j].x)
            tlx = contours_poly[MaxSizeId][j].x;
        if(tly < contours_poly[MaxSizeId][j].y)
            tly = contours_poly[MaxSizeId][j].y;
        if(brx >  contours_poly[MaxSizeId][j].x)
            brx = contours_poly[MaxSizeId][j].x;
        if(bry > contours_poly[MaxSizeId][j].y)
            bry = contours_poly[MaxSizeId][j].y;
    }

    L1 = Point2f(tlx, tly);
    L2 = Point2f(brx, bry);

    ConnerPoint.push_back(L1);
    ConnerPoint.push_back(L2);

    return true;
}




void BodySkeleton::FindFaceConnect(Mat &bw)
{
    Mat labelImage(bw.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    int label = labelImage.at<int>(head.x, head.y);

    if(label > 0)
    {    
        inRange(labelImage, Scalar(label), Scalar(label), bw);
        threshold(bw, bw, 1, 255, THRESH_BINARY);
    }    

}

void findSkeleton(Mat &bw)
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
        if(r->width > FaceRect.width && r->height > FaceRect.height)
        {   
            Point center;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);

            if(center.x > FaceRect.x && center.x < FaceRect.x + FaceRect.width && center.y < FaceRect.y + FaceRect.width*2 )
            {    
                lShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.1), r->y*scale + (r->height-1)*0.9);
                rShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.9), r->y*scale + (r->height-1)*0.9);
                break;
                /*rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                            cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                            Scalar(0, 255, 255), 3, 8, 0);*/
            }
        }    
    }
    //printf("rShoulder: %d, %d   lShoulder: %d, %d\n", body_skeleton.rShoulder.x, 
    //        body_skeleton.rShoulder.y, body_skeleton.lShoulder.x, body_skeleton.lShoulder.y);
}


void BodySkeleton::FindArm(int RightOrLeft)
{
    Mat EDT = FindDistTran(dispMask);
    int fheight = HeadHeight*0.9;
    float Slope = 0;
    //float refValue = EDT.at<unsigned char>(lShoulder.x, lShoulder.y);
    Point elbow;
    if(RightOrLeft == 1)
        elbow = Point(rShoulder);
    else
        elbow = Point(lShoulder);

    Mat proc;
    GaussianBlur(EDT, proc, Size(5, 5), 0);
    //inRange(proc, Scalar(refValue - 10 > 0? refValue - 10 : 2), Scalar(refValue + 3), proc);
    //threshold( proc, proc, 0, 255, THRESH_BINARY|THRESH_OTSU );
    //erode(proc, proc, Mat());
    //imshow("proc", proc);
    //return elbow;

    for(int i = 0; i < 5; i++)
    {
        bool find = false; 
        Point search;
        float TempSlope = 0;

        if(RightOrLeft == 1)
        {
            for(int x = elbow.x + fheight/4 > EDT.cols - 1 ? EDT.cols - 1 : elbow.x + fheight/4; x > (elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4); x--)
            //for(int x = elbow.x - fheight/4 < 0 ? 0 : elbow.x - fheight/4; x < (elbow.x + fheight/4 > EDT.cols-1 ? EDT.cols : elbow.x + fheight/4); x++)
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

        //printf("Slope %f, TempSlope %f\n", Slope, TempSlope);
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
    HandGesture *_hg;
    int FWidth = HeadWidth;
    int nLabels = connectedComponents(SkinSeg, labelImage, 8);
    int label = 0;
    int minD = FWidth;
    int maxD = 0;
    int procD = 0;
    int facelabel = labelImage.at<int>(head.y, head.x);
    vector<Point2f> ConnerPoint;
    //normalize(labelImage, labelImage, 0, 255, NORM_MINMAX, CV_8U);
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
            if(labelImage.at<int>(y,x) != 0 && labelImage.at<int>(y,x) != facelabel)
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
        //Mat element_mask = Mat(Size(5, 5), CV_8UC1, Scalar(1));
        //dilate(mask, mask, element_mask);
        //People.copyTo(handimg, mask);
        //imshow("handimg", handimg);
        //
        double scale = 1.0;
        vector<Rect> hand;
        Mat gray, smallImg( cvRound (SkinSeg.rows/scale), cvRound(SkinSeg.cols/scale), CV_8UC1 );
        cvtColor( handimg, gray, COLOR_BGR2GRAY );
        resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
        equalizeHist( smallImg, smallImg );

        cascade_hand.detectMultiScale( smallImg, hand,
            1.1, 2, 0
            |CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            |CASCADE_SCALE_IMAGE
            ,Size(10, 10) );

        for(vector<Rect>::const_iterator r = hand.begin(); r != hand.end(); r++)
        {
            Hand.x = cvRound((r->x + r->width*0.5)*scale);
            Hand.y = cvRound((r->y + r->height*0.5)*scale);
            if(RightOrLeft)
                rHand = Hand;
            else
                lHand = Hand;
            return;
            /*rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                    cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                    Scalar(0, 255, 255), 3, 8, 0);*/
        }
        //find the most far point of the most close area
        //erode(labelImage, labelImage, element_mask);
        //imshow("hand mask", mask);
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
        //imshow("hand mask", mask);
        
        if(RightOrLeft == 1)
        {
            findContours(mask, _hg->contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
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
                    _hg->eleminateDefects(img, mask);
                }
                bool isHand=_hg->detectIfHand();
                //hg->printGestureInfo(m->src);
                if(isHand){	
                    Moments mo = moments(_hg->contours[_hg->cIdx]);
                    Hand = Point(mo.m10/mo.m00, mo.m01/mo.m00);
                    _hg->getFingerTips(img, mask, Hand, HeadHeight);
                    _hg->drawFingerTips(img);
                    //myDrawContours(m,hg);
		        }
	        }
        }    
        else
        {
            findContours(mask, _hg->contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
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
                    _hg->eleminateDefects(img, mask);
                }
                bool isHand=_hg->detectIfHand();
                //hg->printGestureInfo(m->src);
                if(isHand){	
                    Moments mo = moments(_hg->contours[_hg->cIdx]);
                    Hand = Point(mo.m10/mo.m00, mo.m01/mo.m00);
                    _hg->getFingerTips(img, mask, Hand, HeadHeight);
                    _hg->drawFingerTips(img);
                    //myDrawContours(m,hg);
		        }
	        }
        }    

    }    

    if(RightOrLeft)
        rHand = Hand;
    else
        lHand = Hand;
}    
