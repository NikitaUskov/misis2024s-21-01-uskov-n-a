#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

struct Data {
    cv::Point point;
};

std::vector<Data> readDataFromJson(const std::string& jsonFilePath) {
    cv::FileStorage fs(jsonFilePath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open JSON file: " << jsonFilePath << std::endl;
        return {};
    }

    std::vector<Data> dataVector;

    cv::FileNode objects = fs["objects"];
    for (const auto& obj : objects) {
        Data data;
        data.point.x = static_cast<int>(obj["p"][0]);
        data.point.y = static_cast<int>(obj["p"][1]);

        dataVector.push_back(data);
    }

    return dataVector;
}

cv::Mat sRGB_to_linRGB(cv::Mat& image) {
    cv::Mat lin_image(image.rows, image.cols, CV_32FC3);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            cv::Vec3f lin_pixel;

            for (int k = 0; k < 3; k++) {
                float c = pixel[k] / 255.0f;

                if (c <= 0.04045) {
                    lin_pixel[k] = c / 12.92f;
                } else {
                    lin_pixel[k] = pow((c + 0.055f) / 1.055f, 2.4f);
                }
            }

            lin_image.at<cv::Vec3f>(i, j) = lin_pixel;
        }
    }

    return lin_image;
}


cv::Mat grayWorldWhiteBalance(const cv::Mat& image) {
    cv::Mat result;
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    std::vector<double> means;
    for (int i = 0; i < 3; ++i) {
        auto x = cv::mean(channels[i]);
        means.push_back(x[0]);
    }
    for(int i = 0; i < 3; ++i)
        channels[i] *= means[i]/cv::mean(means)[0];

    cv::merge(channels, result);
    return result;
}

cv::Vec3f getColorInLinRGB(const cv::Mat& image, const cv::Point& point) {
    cv::Vec3b sRGB_color = image.at<cv::Vec3b>(point);
    cv::Vec3f linRGB_color;

    for (int i = 0; i < 3; i++) {
        float c = sRGB_color[i] / 255.0f;

        if (c <= 0.04045) {
            linRGB_color[i] = c / 12.92f;
        } else {
            linRGB_color[i] = std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
    }

    return linRGB_color;
}

void coutMetrics(const cv::Mat& image1, const cv::Mat& image2, const std::vector<Data>& data, std::string name) {
    float metric1 = 0.0f;
    float metric2 = 0.0f;
    for (const auto& d : data) {
        cv::Vec3f color1 = getColorInLinRGB(image1, d.point);
        cv::Vec3f color2 = getColorInLinRGB(image2, d.point);
        float norm = cv::norm(color1 - color2);
        metric1 += norm;
        metric2 += norm * norm;
    }

    metric1 /= data.size();
    metric2 /= data.size();

    std::cout << name << std::endl;
    std::cout << "Mean: " << metric1 << std::endl;
    std::cout << "RMS: " << std::sqrt(metric2) << std::endl;
}


int main() {
    cv::Mat image = cv::imread("/home/art4m/misis/misis2024s-21-02-eroshkin-a-t/prj.lab/lab09/src/pic.png");
    std::vector<Data> data = readDataFromJson("/home/art4m/misis/misis2024s-21-02-eroshkin-a-t/prj.lab/lab09/data.json");
    cv::Mat lin_image = sRGB_to_linRGB(image);
    cv::Mat white_balanced_image = grayWorldWhiteBalance(image);
    cv::Mat contrasted_image = autoContrastColor(image, 0.1, 0.9);

    coutMetrics(image, white_balanced_image, data, "White balance");
    coutMetrics(image, contrasted_image, data, "Contrast");

    cv::imshow("Original", image);
    cv::imshow("White Balanced", white_balanced_image);
    cv::imwrite("White.png", white_balanced_image);
    cv::imshow("Contrasted", contrasted_image);
    cv::imwrite("Contrasted.jpg", contrasted_image);
    cv::waitKey(0);

    return 0;
}
