#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

// Convert sRGB to Linear RGB
cv::Mat sRGBtoLinRGB(const cv::Mat& img) {
    cv::Mat temp;
    img.convertTo(temp, CV_32F, 1.0 / 255);
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            for (int k = 0; k < 3; k++) {
                float& value = temp.at<cv::Vec3f>(i, j)[k];
                if (value <= 0.04045) {
                    value /= 12.92;
                }
                else {
                    value = std::pow((value + 0.055) / 1.055, 2.4);
                }
            }
        }
    }
    return temp;
}

// Find the norm of a 3-channel pixel
double find_norm(const cv::Vec3f& pixel) {
    return std::sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);
}

// Find the cosine of the angle between two 3-channel pixels
double find_cos(const cv::Vec3f& pixel1, const cv::Vec3f& pixel2) {
    double product = pixel1[0] * pixel2[0] + pixel1[1] * pixel2[1] + pixel1[2] * pixel2[2];
    double norm1 = find_norm(pixel1);
    double norm2 = find_norm(pixel2);
    if (norm1 == 0 || norm2 == 0) {
        return 0;
    }
    return product / (norm1 * norm2);
}

// Change the plane of a 3-channel vector
cv::Vec2f change_plane(const cv::Vec3f& vec) {
    cv::Vec3f vec_i(-0.25, -0.25, 0.5);
    cv::Vec3f vec_j(-0.5, 0.5, 0);
    double cos_i = find_cos(vec, vec_i);
    double cos_j = find_cos(vec, vec_j);
    double pr_i = find_norm(vec) * cos_i * std::sqrt(2) / 3;
    double pr_j = find_norm(vec) * cos_j * std::sqrt(2) / 3;
    pr_i = std::sqrt(3) / 3 - pr_i;
    pr_j = 0.5 + pr_j;
    return cv::Vec2f(pr_i, pr_j);
}

// Prepare the projection plane
cv::Mat prepare_proj_plane() {
    cv::Mat plane = cv::Mat::zeros(230, 260, CV_8UC3);
    plane += cv::Scalar(100, 100, 100);
    plane.at<cv::Vec3b>(4, 129) = cv::Vec3b(0, 0, 255); // Offset by 4 and 2
    plane.at<cv::Vec3b>(4, 130) = cv::Vec3b(0, 0, 255);
    plane.at<cv::Vec3b>(226, 2) = cv::Vec3b(255, 0, 0);
    plane.at<cv::Vec3b>(226, 257) = cv::Vec3b(0, 255, 0);
    for (int i = 1; i < 256; i++) {
        plane.at<cv::Vec3b>(static_cast<int>(4 + i * 0.87), static_cast<int>(130 + i / 2)) = cv::Vec3b(0, i, 255 - i);
        plane.at<cv::Vec3b>(static_cast<int>(4 + i * 0.87), static_cast<int>(129 - i / 2)) = cv::Vec3b(i, 0, 255 - i);
        plane.at<cv::Vec3b>(226, 2 + i) = cv::Vec3b(255 - i, i, 0);
    }
    return plane;
}

// Project the input image onto the plane
cv::Mat Projection(const cv::Mat& input) {
    cv::Mat temp = sRGBtoLinRGB(input);
    cv::Mat proj = prepare_proj_plane();
    for (int j = 0; j < temp.rows; j++) {
        for (int i = 0; i < temp.cols; i++) {
            cv::Vec3f pixel = temp.at<cv::Vec3f>(j, i);
            double cos = find_cos(pixel, cv::Vec3f(1, 1, 1));
            double a = std::sqrt(3) / (2 * cos * find_norm(pixel));
            cv::Vec3f new_pixel = a * pixel - cv::Vec3f(0.5, 0.5, 0.5);
            cv::Vec2f new_pixel_pr = change_plane(new_pixel);
            int x = static_cast<int>(new_pixel_pr[0] * 256 + 4);
            int y = static_cast<int>(new_pixel_pr[1] * 256 + 2);
            if (x >= 0 && x < proj.rows && y >= 0 && y < proj.cols) {
                proj.at<cv::Vec3b>(x, y) += cv::Vec3b(1, 1, 1);
            }
        }
    }
    return proj;
}

int main() {
    cv::Mat img = cv::imread("C://opencv/lab08/Granfdfather.jpg");
    if (img.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    cv::Mat proj = Projection(img);
    cv::imshow("plane", proj);
    cv::imwrite("4.jpg", proj);
    cv::waitKey(0);
    return 0;
}
