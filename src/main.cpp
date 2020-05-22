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
#define CRITERIA_1 8
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
    int
        srand(time(NULL));
    unsigned int id = 0;
    for (unsigned int i = 0; i < SPLITS; ++i)
    {
        for (unsigned int j = 0; j < SPLITS; ++j)
        {
            cv::Point2i seed(i * cell_width + rand() % cell_width,
                             j * cell_height + rand() % cell_height);
            Region r(id++);
            r.AddPixel(seed, image.at<cv::Vec3b>(seed));
            regions.push_back(r);
        }
    }
}

// void seed_placing(cv::Mat &image, std::vector<Region> &regions)
// {
//     unsigned int cell_size = SPLITS;

//     srand(time(NULL));
//     for (unsigned int i = 0; i * cell_size < image.size().width + cell_size; ++i)
//     {
//         for (unsigned int j = 0; j * cell_size < image.size().height + cell_size; ++j)
//         {
//             cv::Point2i seed(i * cell_size + rand() % cell_size,
//                              j * cell_size + rand() % cell_size);
//             Region r;
//             r.AddPixel(seed, image.at<cv::Vec3b>(seed));
//             regions.push_back(r);
//         }
//     }
// }
void set_image_avg_color(cv::Mat &image, std::vector<Region> &regions)
{
    for (auto &region : regions)
    {
        cv::Vec3b color = region.CalcAvg();
        for (auto p : region.GetPixels())
            image.at<cv::Vec3b>(p) = color;
        for (auto p : region.GetBorderPixels())
            image.at<cv::Vec3b>(p) = cv::Vec3b();
    }
}

void set_image_color(cv::Mat &image, std::vector<Region> &regions,bool * ControleBool)
{
    for (auto &region : regions)
    {
        if(!ControleBool[region._id])
        {
            RNG rng;
            int r = rng.uniform(0, 256);
            int g = rng.uniform(0, 256);
            int b = rng.uniform(0, 256);
            cv::Vec3b color = cv::Vec3b(b,g,r);
            for (auto p : region.GetPixels())
                image.at<cv::Vec3b>(p) = color;
        }
    }
}

void region_growing(cv::Mat &image, std::vector<Region> &regions)
{
    
    unsigned int regions_nb = regions.size();
    unsigned int width = image.size().width;
    unsigned int height = image.size().height;

    unsigned int pixels[image.size().width][image.size().height];
    for (int i = 0; i < image.size().width; ++i)
    {
        for (int j = 0; j < image.size().height; ++j)
            pixels[i][j] = -1;
    }
    int marked_count = 0;
    for (auto &region : regions)
    {
        cv::Point2i p = region.GetPixels().front();
        pixels[p.x][p.y] = region._id;
        region.AddMarkedPixel(p);
        marked_count++;
    }
    int count = image.size().area();
    while (marked_count != 0)
    {
        for (auto &region : regions)
        {
            if (region.MarkedPixelEmpty())
                continue;
            cv::Point2i p = region.GetMarkedPixel();
            marked_count--;
            cv::Vec3i p_color = (cv::Vec3i)image.at<cv::Vec3b>(p);
            for (int ii = -1; ii < 2; ++ii)
            {
                for (int jj = -1; jj < 2; ++jj)
                {
                    cv::Point2i pp(p.x + ii, p.y + jj);
                    if (!(pp.x < 0 || pp.x >= width || pp.y < 0 || pp.y >= height))
                    {
                        if (pixels[pp.x][pp.y] == -1)
                        {
                            cv::Vec3b pp_color = image.at<cv::Vec3b>(pp);
                            float distance = cv::norm((cv::Vec3i)pp_color, (cv::Vec3i)p_color);

                            if (distance < CRITERIA_1)
                            {
                                region.AddPixel(pp, pp_color);
                                region.AddMarkedPixel(pp);
                                marked_count++;
                                pixels[pp.x][pp.y] = region._id;
                            }
                            else
                            {
                                region.AddBorderPixel(pp);
                            }
                        }
                        else
                        {
                            if (pixels[pp.x][pp.y] != region._id)
                            {
                                region.AddBorderPixel(pp);
                                region.AddNeighbour(pixels[pp.x][pp.y]);
                            }
                        }
                    }
                }
            }
        }
    }
}

void region_merging(cv::Mat &image, std::vector<Region> &regions)
{
    cv::Mat testimg(image.size(), image.type(), cv::Scalar(0, 0, 0));
    float distance;
    bool ControleBool[regions.size()] = {0};
    unsigned int count = 0;
    for (auto &region : regions)
    {
        bool change = true;
        while (change && !ControleBool[region._id])
        {
            change = false;
            std::set<unsigned int> neighbours = region.GetNeighbours();
            for (auto n = neighbours.begin(); n != neighbours.end(); n++)
            {
                Region neighbour = regions.at(*n);
                distance = cv::norm(region.CalcAvg(), neighbour.CalcAvg());
                if (distance < CRITERIA_1 && !ControleBool[neighbour._id])
                {
                    change = true;
                    ControleBool[*n] = 1;
                    std::set<unsigned int> neighbours_of_absorbed = region.AbsorbRegion(neighbour);
                    for (auto id : neighbours_of_absorbed)
                    {
                        if (id != region._id)
                        {
                            Region nu = regions.at(id);
                            nu.ChangeNeighbour(*n, region._id);
                        }
                    }
                }
            }
        }
    } 
    }
    set_image_color(testimg, regions,controleBool);
    cv::resize(testimg, testimg, cv::Size(), 2, 2, 0);

    cv::imshow("image", testimg);
}

void segmentation(const cv::Mat &input_image, cv::Mat &output_image)
{
    std::vector<Region> regions;
    preprocessing(input_image, output_image);
    seed_placing(output_image, regions);
    region_growing(output_image, regions);
    region_merging(output_image, regions);
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
