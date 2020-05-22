#include <opencv2/core/types.hpp>

#include <queue>
#include <set>

class Region
{
public:
    Region(unsigned int id) : _marked_pixels(), _border_pixels(), _pixels(), _neighbours(), _id(id), _avg() {}
    void AddPixel(cv::Point2i pixel_pos, cv::Vec3b pixel_color);
    std::set<unsigned int> AbsorbRegion(Region &r);
    std::vector<cv::Point2i> GetPixels();
    cv::Point2i GetMarkedPixel();
    bool MarkedPixelEmpty();
    std::deque<cv::Point2i> &GetBorderPixels();
    void AddMarkedPixel(cv::Point2i pixel);
    void AddBorderPixel(cv::Point2i pixel);
    void AddNeighbour(unsigned int id);
    void ChangeNeighbour(unsigned int old_id, unsigned int new_id);
    std::set<unsigned int> &GetNeighbours();
    void CalcAvg();
    cv::Vec3b GetColor();
    int GetCount();
    unsigned int _id;

protected:
    int GetSum(int channel);

private:
    std::deque<cv::Point2i> _marked_pixels;
    std::deque<cv::Point2i> _border_pixels;
    std::vector<cv::Point2i> _pixels;
    std::set<unsigned int> _neighbours;
    std::set<unsigned int> _old_neighbours;
    int _sum[3] = {0};
    cv::Vec3b _avg;
    int _count = 0;
};
