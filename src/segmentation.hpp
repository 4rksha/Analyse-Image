#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

#include <queue>

class Region
{
public:
    Region() : _marked_pixels(), _border_pixels(), _pixels() {};
    void AddPixel(cv::Point2i pixel_pos, cv::Vec3b pixel_color);
    void AbsorbRegion(Region &r);
    void CalcAvg();
    std::vector<cv::Point2i> &GetPixels();
    cv::Point2i &GetMarkedPixel();
    std::deque<cv::Point2i> &GetBorderPixels();
    void AddMarkedPixel(cv::Point2i &pixel);
    void AddBorderPixel(cv::Point2i &pixel);

protected:
    int GetSum(int channel);

private:
    std::deque<cv::Point2i> _marked_pixels;
    std::deque<cv::Point2i> _border_pixels;
    std::vector<cv::Point2i> _pixels;
    int _sum[3];
    cv::Vec3b _avg;
};
