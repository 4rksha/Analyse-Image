#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;


vector<Scalar> setRgb()
{
    RNG rng;
	vector<Scalar> colors;
    for(int i = 0; i < 2000; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
	return colors;
}

Mat segmentation_simple(Mat frame1,vector<Scalar> colors,int nbcluster)
{
	Mat label;
    TermCriteria criteria = TermCriteria((TermCriteria::MAX_ITER) + (TermCriteria::EPS), 10, 1.0);
	Mat center;
	Mat frame2;
    frame1.convertTo(frame2, CV_32F, 1/255.0);
	frame2 = frame2.reshape(1,frame2.total());
    kmeans(frame2,nbcluster,label,criteria,3,KMEANS_PP_CENTERS,center);
    center = center.reshape(3,center.rows);
    frame2 = frame2.reshape(3,frame2.rows);
    Point3f *p = frame2.ptr<Point3f>();
    for (int i=0; i<frame2.rows; i++) {
       int center_id = label.at<int>(i);
       Point3f point;
       point.x= colors[center_id].conj()[0];
       point.y= -colors[center_id].conj()[1];
       point.z= -colors[center_id].conj()[2];
       p[i] = point;//center.at<Point3f>(center_id)*255;
    }
    frame1 = frame2.reshape(3, frame1.rows);
    frame1.convertTo(frame1, CV_8U);
	return frame1;
}

Mat Wattershed(Mat src,vector<Scalar> colors)
{
    for ( int i = 0; i < src.rows; i++ ) {
        for ( int j = 0; j < src.cols; j++ ) {
            if ( src.at<Vec3b>(i, j) == Vec3b(255,255,255) )
            {
                src.at<Vec3b>(i, j)[0] = 0;
                src.at<Vec3b>(i, j)[1] = 0;
                src.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
    Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1);
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    threshold(dist, dist, 0.2, 1.0, THRESH_BINARY);
    Mat kernel1 = Mat::ones(5, 5, CV_8U);
    dilate(dist, dist, kernel1);
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }
    circle(markers, Point(5,5), 3, Scalar(255), -1);
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
	vector<Vec3b> colors2;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = colors[i].conj()[0];
        int g = - colors[i].conj()[1];
        int r = -colors[i].conj()[2];
        colors2.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors2[index-1];
            }
        }
    }
    return dst;
}

int main()
{
	Mat frame1,gray1;
	Mat frameValue,grayValue;
	VideoCapture capture("./data/video1.mp4");
    if(!capture.isOpened())
    {
        cout << "Could not open reference ";
        cout << endl;
        return -1;
    }
    capture.read(frame1);
	cvtColor(frame1, gray1, COLOR_BGR2GRAY);
	while ( capture.read(frameValue))
	{
		if( frameValue.empty() )
        {
            cout << "--(!) No captured frameValue -- Break!\n";
            break;
        }
		cvtColor(frameValue, grayValue, COLOR_BGR2GRAY);
		vector<Scalar> colors = setRgb();
		Mat frameValue2=segmentation_simple(frameValue,colors, 32);
		Mat frameValue3=segmentation_simple(frameValue,colors, 16);
		Mat frameValue4=segmentation_simple(frameValue,colors, 8);
		/*Mat frameValue3=Wattershed(frameValue2,colors);
		Mat frameValue4=Wattershed(frameValue,colors);*/
		imshow("image",frameValue);
		imshow("image2",frameValue2);
		imshow("image3",frameValue3);
		imshow("image4",frameValue4);
		if( waitKey(10) == 27)
        {
            break; // escape
        }
	}
	return 0;
}
