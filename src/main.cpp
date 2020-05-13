#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <queue>
#include "segmentation.hpp"
#include <cmath>

#define SPLITS 20
#define CRITERIA_1 50

void preprocessing(const cv::Mat &input_image, cv::Mat &image)
{
    cv::bilateralFilter(input_image, image, 11, 80, 60);
    //cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    // cv::Mat mask;
    // mask.create(image.size(), image.type());
    // mask = cv::Scalar::all(0);
    // cv::threshold(image, mask, 250, 255, cv::ThresholdTypes::THRESH_BINARY_INV);
    // cv::Mat tmp;
    cv::equalizeHist(image, image);
    // image.copyTo(tmp, mask);
    // image = tmp.clone();
}
void seed_placing(cv::Mat &output_image, std::vector<cv::Point2i> &seeds)
{
    unsigned int cell_width = output_image.size().width / SPLITS;
    unsigned int cell_height = output_image.size().height / SPLITS;

    srand(time(NULL));
    for (unsigned int i = 0; i < SPLITS; ++i)
    {
        for (unsigned int j = 0; j < SPLITS; ++j)
        {
            cv::Point2f p(i * cell_width + rand() % cell_width,
                          j * cell_height + rand() % cell_height);
            seeds.push_back(p);
        }
    }
    std::cout << rand() << std::endl;
}

void region_growing(cv::Mat &image, std::vector<cv::Point2i> &seeds)
{
    cv::Mat testimg(image.size().height, image.size().width, image.type(), cv::Scalar(0, 0, 0));
    unsigned int regions_nb = seeds.size();
    unsigned int width = image.size().width;
    unsigned int height = image.size().height;

    Pixel **pixels = new Pixel *[image.size().width];
    for (int i = 0; i < width; ++i)
    {
        pixels[i] = new Pixel[height];
    }
    std::deque<cv::Point2i> marked[regions_nb];
    std::deque<cv::Point2i> borders[regions_nb];
    for (unsigned int i = 0; i < seeds.size(); ++i)
    {
        pixels[seeds[i].x][seeds[i].y].mark = true;
        pixels[seeds[i].x][seeds[i].y].region = i;
        marked[i].push_back(seeds[i]);
    }
    int count = image.size().area();

    while (count != 0)
    {
        int testEmpty = regions_nb;
        for (unsigned int i = 0; i < regions_nb; ++i)
        {
            if (marked[i].empty())
            {
                testEmpty--;
                if (!testEmpty)
                {
                    count = 0;
                }
                continue;
            }
            cv::Point2i p = marked[i].front();

            marked[i].pop_front();
            count--;
            bool hasborder = false;
            uchar intensity = image.at<uchar>(p);
            for (int ii = -1; ii < 2; ++ii)
            {
                for (int jj = -1; jj < 2; ++jj)
                {
                    cv::Point2i pp(p.x + ii, p.y + jj);
                    if (pp.x < 0 || pp.x >= width || pp.y < 0 || pp.y >= height)
                    {
                        borders[i].push_back(p);
                    }
                    else if (!pixels[pp.x][pp.y].mark)
                    {
                        if (std::abs(image.at<uchar>(pp) - intensity) < CRITERIA_1)
                        {
                            pixels[pp.x][pp.y].mark = true;
                            pixels[pp.x][pp.y].region = i;
                            //testimg.at<uchar>(pp) = 255; //white
                            marked[i].push_back(pp);
                        }
                        else
                        {
                            borders[i].push_back(p);
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < regions_nb; ++i)
    {
        for(auto it=borders[i].begin(); it!=borders[i].end();++it) {

            testimg.at<uchar>(*it) = 255; //white
        }
    }
    
}

void segmentation(const cv::Mat &input_image, cv::Mat &output_image)
{
    std::vector<cv::Point2i> seeds;
    preprocessing(input_image, output_image);
    //seed_placing(output_image, seeds);

    //region_growing(output_image, seeds);
    cv::resize(output_image, output_image, cv::Size(), 3, 3);
    cv::imshow("image", output_image);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << " Usage: main FileToLoadAndDisplay" << std::endl;
        return -1;
    }
    std::string filename = argv[1];
    if (filename.find(".jpg") != std::string::npos || filename.find(".png") != std::string::npos)
    {
        cv::Mat input_image;
        input_image = cv::imread(argv[1], cv::IMREAD_COLOR);

        if (!input_image.data)
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        cv::Mat segmented_image;
        segmentation(input_image, segmented_image);
        //cv::imshow("image", segmented_image);
        cv::waitKey(0);
    }
    else
    {
        std::cout << " Incorrect file format. \n Accepted format : \n- Image: .png, .jpg\n- Video: .mp4, .avi \n Exiting." << std::endl;
    }
    return 0;
}
