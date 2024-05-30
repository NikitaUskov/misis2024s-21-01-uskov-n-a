#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

cv::Vec3f normalize(const cv::Vec3f& v) {
    return v / cv::norm(v, cv::NORM_L2);
}

std::vector<cv::Point2f> projectColors(const cv::Mat& src) {
    std::vector<cv::Point2f> result{};
    const cv::Vec3f ox{ normalize({ -1.0f, 1.0f, 0.0f }) };
    const cv::Vec3f oy{ normalize({ -1.0f, -1.0f, 2.0f }) };
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            const auto& color{ src.at<cv::Vec3f>(i, j) };
            const float alpha = 1.5f / (color.dot({ 1.0f, 1.0f, 1.0f }));
            if (std::isinf(alpha)) {
                continue;
            }
            const auto proj{ alpha * color - cv::Vec3f{ 0.5f, 0.5f, 0.5f } };
            result.emplace_back(proj.dot(ox), proj.dot(oy));
        }
    }
    return result;
}

cv::Mat getProj(const std::vector<cv::Point2f>& points, const int size) {
    const int height = static_cast<int>(std::sqrt(3.0) / 2 * size);
    cv::Mat result{ cv::Mat::zeros(height, size, CV_16UC1) };
    int radius = 0.0; // Радиус круга для "жирных" точек

    for (const auto& p : points) {
       const int x = static_cast<int>((p.x + 0.75f * std::sqrt(2.0f)) / (1.5f * std::sqrt(2.0f)) * static_cast<float>(size - 1));
       const int y = static_cast<int>((p.y + 0.25f * std::sqrt(6.0f)) / (0.75f * std::sqrt(6.0f)) * static_cast<float>(height - 1));
       cv::circle(result, cv::Point(x, height - y - 1), radius, cv::Scalar(1), -1); // рисуем круг с радиусом
    }

    double max = 0.0;
    cv::minMaxLoc(result, nullptr, &max);
    result.convertTo(result, CV_32FC1, 1.0 / max);
    return result;
}

int main(int argc, char* argv[]) {

    std::string input = "C:/opencv/lab08/Радуга.jpg";
    std::string output = "C:/Users/79194/OneDrive - НИТУ МИСиС/Рисунки/Рандом фотки/СобакаH.jpg";
    int size = 256;
    if (argc >= 4) {
        input = argv[1];
        output = argv[2];
        size = std::stoi(argv[3]);
    }

    cv::Mat img{ cv::imread(input) };
    img.convertTo(img, CV_32FC3, 1.0f / std::numeric_limits<uchar>::max());

    const auto points{ projectColors(img) };

    cv::Mat result{ getProj(points, size) };
    result.convertTo(result, CV_8UC1, std::numeric_limits<uchar>::max());

    cv::imwrite(output, result);
    cv::imshow("orig", img);
    cv::imshow("result", result);
    cv::imwrite("Result.png", result);
    cv::waitKey(0);
}
