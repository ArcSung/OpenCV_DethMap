#include "skeleton.hpp"

using namespace cv;
using namespace std;

double CalcuDistance(Point P1, Point P2)
{
    return norm(P1 - P2);
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
        Size(10, 10) );
    for( vector<Rect>::const_iterator r = upbody.begin(); r != upbody.end(); r++, i++ )
    {
        if(r->width > FaceRect.width && r->height > FaceRect.height)
        {   
            Point center;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);

            if(center.x > FaceRect.x && center.x < FaceRect.x + FaceRect.width && center.y < FaceRect.y + FaceRect.width*2 )
            {    
                printf("find upbody\n");
                body_skeleton.rShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.1), r->y*scale + (r->height-1)*0.9);
                body_skeleton.lShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.9), r->y*scale + (r->height-1)*0.9);
                break;
                /*rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                            cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                            Scalar(0, 255, 255), 3, 8, 0);*/
            }
        }    
    }
    printf("rShoulder: %d, %d   lShoulder: %d, %d\n", body_skeleton.rShoulder.x, 
            body_skeleton.rShoulder.y, body_skeleton.lShoulder.x, body_skeleton.lShoulder.y);
}


Point findArm(Mat EDT, Point lShoulder, int fheight, int findLeftelbow)
{
    float Slope = 0;
    float refValue = EDT.at<unsigned char>(lShoulder.x, lShoulder.y);
    Point elbow = lShoulder;
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

Point findHand(Mat Skin, Mat People, CascadeClassifier& cascade_hand, Point rElbow, Point FacePoint, int FWidth)
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

    /*Mat temp = Mat(mask.size(), mask.type());
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

    for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
    {
        if (it->size()>50)
            drawContours(temp, contours, 0, Scalar(255), CV_FILLED, 8, hierarchy);
        ++it;
    }*/

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
            if(labelImage.at<int>(y,x)  == label)
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
