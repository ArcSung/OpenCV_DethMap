// Preprocess.cpp

#include "Preprocess.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////
void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh) {
    imgGrayscale = extractValue(imgOriginal);                           // extract value channel only from original image to get imgGrayscale

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // maximize contrast with top hat and black hat

    cv::Mat imgBlurred;

    cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          // gaussian blur

                // call adaptive threshold to get imgThresh
    cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat extractValue(cv::Mat &imgOriginal) {
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;
    cv::Mat imgValue;

    cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

    cv::split(imgHSV, vectorOfHSVImages);

    imgValue = vectorOfHSVImages[2];

    return(imgValue);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat maximizeContrast(cv::Mat &imgGrayscale) {
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));

    cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);
    cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

    imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

    return(imgGrayscalePlusTopHatMinusBlackHat);
}

Mat CalcuEDT(Mat DT, Point ref)
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
    }
    return EDT;
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

double CalcuDistance(Point P1, Point P2)
{
    return norm(P1 - P2);
}    

Mat FindDistTran(Mat bw)
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

int findBiggestContour(vector<vector<Point> > contours){
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

float getAngle(Point s, Point f, Point e){
	float l1 = CalcuDistance(f,s);
	float l2 = CalcuDistance (f,e);
	float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
	float angle = acos(dot/(l1*l2));
	angle=angle*180/PI;
	return angle;
}
