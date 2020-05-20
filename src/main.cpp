#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include "segmentation.hpp"
#include <cmath>

#define SPLITS 20
#define CRITERIA_1 24
void preprocessing(const cv::Mat &input_image, cv::Mat &image)
{
    //cv::cvtColor(input_image, image, cv::COLOR_BGR2GRAY);
    //cv::bilateralFilter(input_image, image, 5, 80, 80);
    image = input_image.clone();
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
}

void region_growing(cv::Mat &image, std::vector<cv::Point2i> &seeds)
{
    cv::Mat testimg(image.size(), image.type(), cv::Scalar(0, 0, 0));
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
                        float distance = cv::norm((cv::Vec3i) image.at<cv::Vec3b>(pp),(cv::Vec3i) image.at<cv::Vec3b>(p));

                        if (distance < CRITERIA_1)
                        {
                            pixels[pp.x][pp.y].mark = true;
                            pixels[pp.x][pp.y].region = i;
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

    cv::Vec3b colors[regions_nb];
    for (unsigned int i = 0; i < regions_nb; ++i)
    {
        colors[i][0] = 70 + rand() % 185;
        colors[i][1] = 70 + rand() % 185;
        colors[i][2] = 70 + rand() % 185;
    }
    for (unsigned int j = 0; j < image.size().width; ++j)
    {
        for (unsigned int k = 0; k < image.size().height; ++k)
        {
            testimg.at<cv::Vec3b>(cv::Point2i(j, k)) = colors[pixels[j][k].region];
        }
    }

    for (unsigned int i = 0; i < regions_nb; ++i) {
        for (auto p : borders[i]) {
            testimg.at<cv::Vec3b>(p) = cv::Vec3b();
        }
    }
    cv::resize(testimg, testimg, cv::Size(), 3, 3);

    cv::imshow("image", testimg);
}

void segmentation(const cv::Mat &input_image, cv::Mat &output_image)
{
    std::vector<cv::Point2i> seeds;
    preprocessing(input_image, output_image);
    seed_placing(output_image, seeds);
    region_growing(output_image, seeds);
    //cv::resize(output_image, output_image, cv::Size(), 3, 3);
    //cv::imshow("image", output_image);
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
        cv::waitKey(0);
    }
    else
    {
        std::cout << " Incorrect file format. \n Accepted format : \n- Image: .png, .jpg\n- Video: .mp4, .avi \n Exiting." << std::endl;
    }
    return 0;
}
