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
#define CRITERIA_1 10
void preprocessing(const cv::Mat &input_image, cv::Mat &image)
{
    //cv::cvtColor(input_image, image, cv::COLOR_BGR2GRAY);
    cv::bilateralFilter(input_image, image, 8, 50, 30);
    if (image.size().height > 1000 || image.size().width > 1000)
        cv::resize(image, image, cv::Size(), .5, .5);

    //cv::fastNlMeansDenoisingColored(image, image);
    //cv::addWeighted(input_image, 0.5, image, 0.2, 0, image);
    //image = input_image.clone();
}
void seed_placing(cv::Mat &image, std::vector<Region> &regions)
{
    unsigned int cell_width = image.size().width / SPLITS;
    unsigned int cell_height = image.size().height / SPLITS;

    srand(time(NULL));
    for (unsigned int i = 0; i < SPLITS; ++i)
    {
        for (unsigned int j = 0; j < SPLITS; ++j)
        {
            cv::Point2i seed(i * cell_width + rand() % cell_width,
                             j * cell_height + rand() % cell_height);
            Region r;
            r.AddPixel(seed, image.at<cv::Vec3b>(seed));
            regions.push_back(r);
        }
    }
}

void region_growing(cv::Mat &image, std::vector<Region> &regions)
{
    cv::Mat testimg(image.size(), image.type(), cv::Scalar(0, 0, 0));
    unsigned int regions_nb = regions.size();
    unsigned int width = image.size().width;
    unsigned int height = image.size().height;

    bool pixels[image.size().width][image.size().height];
    int marked_count = 0;
    std::deque<cv::Point2i> borders[regions_nb];
    for (auto region : regions)
    {
        cv::Point2i p = region.GetPixels().front();
        pixels[p.x][p.y] = true;
        region.AddMarkedPixel(p);
        marked_count++;
    }
    int count = image.size().area();

    while (marked_count != 0)
    {
        for (auto region : regions)
        {
            cv::Point2i p = region.GetMarkedPixel();
            std::cout << p << std::endl;
            marked_count--;
            cv::Vec3i p_color = (cv::Vec3i)image.at<cv::Vec3b>(p);
            for (int ii = -1; ii < 2; ++ii)
            {
                for (int jj = -1; jj < 2; ++jj)
                {
                    cv::Point2i pp(p.x + ii, p.y + jj);
                    if (!(pp.x < 0 || pp.x >= width || pp.y < 0 || pp.y >= height) && !pixels[pp.x][pp.y])
                    {
                        cv::Vec3i pp_color = image.at<cv::Vec3b>(pp);
                        float distance = cv::norm((cv::Vec3i)pp_color, (cv::Vec3i)p_color);

                        region.AddPixel(pp, pp_color);
                        pixels[pp.x][pp.y] = true;
                        if (distance < CRITERIA_1)
                        {
                            region.AddMarkedPixel(pp);
                            marked_count++;
                        }
                        else
                        {
                            region.AddBorderPixel(pp);
                        }
                    }
                }
            }
        }
    }

    // cv::Vec3b colors[regions_nb];
    // for (unsigned int i = 0; i < regions_nb; ++i)
    // {
    //     colors[i][0] = 70 + rand() % 185;
    //     colors[i][1] = 70 + rand() % 185;
    //     colors[i][2] = 70 + rand() % 185;
    // }
    // for (unsigned int j = 0; j < image.size().width; ++j)
    // {
    //     for (unsigned int k = 0; k < image.size().height; ++k)
    //     {
    //         image.at<cv::Vec3b>(cv::Point2i(j, k)) = colors[pixels[j][k].region];
    //     }
    // }

    // for (unsigned int i = 0; i < regions_nb; ++i)
    // {
    //     for (auto p : borders[i])
    //     {
    //         image.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 0);
    //     }
    // }
    cv::resize(image, image, cv::Size(), 3, 3);

    cv::imshow("image", image);
}

// void region_merging(cv::Mat &image, Pixel** pixels) {

// }

void segmentation(const cv::Mat &input_image, cv::Mat &output_image)
{
    std::vector<Region> regions;
    preprocessing(input_image, output_image);
    seed_placing(output_image, regions);
    region_growing(output_image, regions);
    //region_merging(output_image, pixels);
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
