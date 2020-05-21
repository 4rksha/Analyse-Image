#include <iostream>
#include "segmentation.hpp"

void Region::AddPixel(cv::Point2i pixel_pos, cv::Vec3b pixel_color)
{
    _pixels.push_back(pixel_pos);
    for (unsigned int i = 0; i < 3; ++i)
    {
        _sum[i] += pixel_color.val[i];
    }
}

void Region::AbsorbRegion(Region &r)
{
    _pixels.insert(_pixels.end(), r.GetPixels().begin(), r.GetPixels().end());
    for (unsigned int i = 0; i < 3; ++i)
    {
        _sum[i] += r.GetSum(i);
    }
}

void Region::CalcAvg()
{
    int size = _pixels.size();
    for (unsigned int i = 0; i < 3; ++i)
    {
        if (_sum[i] / size > 255)
            std::cout << "@@@@@" << _sum[i] / size << std::endl;
        _avg.val[i] = _sum[i] / size;
    }
}

std::vector<cv::Point2i> &Region::GetPixels()
{
    return _pixels;
}

int Region::GetSum(int channel)
{
    return _sum[channel];
}

cv::Point2i &Region::GetMarkedPixel()
{
    cv::Point2i &p = _marked_pixels.front();
    std::cout << "p :" << p << std::endl;
    _marked_pixels.pop_front();
    return p;
}

void Region::AddMarkedPixel(cv::Point2i &pixel)
{
    std::cout << "p :" << pixel << std::endl;

    _marked_pixels.push_back(pixel);
}

void Region::AddBorderPixel(cv::Point2i &pixel)
{
    _border_pixels.push_back(pixel);
}

std::deque<cv::Point2i> &Region::GetBorderPixels()
{
    return _border_pixels;
}
