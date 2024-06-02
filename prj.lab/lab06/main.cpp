#include <iostream>
#include <opencv2/opencv.hpp>

// ���������� ����������
cv::Mat src, binaryImg, detectedImg;
int thresholdValue = 128;
std::vector<cv::Vec3f> groundTruths;

// ������� ��� ����������� �����������
void binaryThreshold(const cv::Mat& src, cv::Mat& dst, int threshold) {
    cv::threshold(src, dst, threshold, 255, cv::THRESH_BINARY);
}

// ������� ��� ������ �������� �� IoU
void evaluateDetections(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) {
    int TP = 0, FP = 0, FN = 0;

    for (const auto& truth : groundTruths) {
        bool detected = false;
        cv::Point truthCenter(cvRound(truth[0]), cvRound(truth[1]));
        int truthRadius = cvRound(truth[2]);

        for (const auto& detection : detections) {
            cv::Point detectCenter(cvRound(detection[0]), cvRound(detection[1]));
            int detectRadius = cvRound(detection[2]);

            double distance = cv::norm(truthCenter - detectCenter);
            double radiusSum = truthRadius + detectRadius;
            double iou = (distance <= radiusSum) ? 1.0 : 0.0; // �������� ���������� IoU

            if (iou >= iouThreshold) {
                TP++;
                detected = true;
                break;
            }
        }
        if (!detected) FN++;
    }

    FP = detections.size() - TP;

    std::cout << "TP: " << TP << std::endl;
    std::cout << "FP: " << FP << std::endl;
    std::cout << "FN: " << FN << std::endl;
}

// �������� ����� ��� ��������
void onTrackbar(int, void*) {
    // ����������� �����������
    binaryThreshold(src, binaryImg, thresholdValue);

    // �������������� ������
    std::vector<cv::Vec3f> detectedCircles;
    cv::HoughCircles(binaryImg, detectedCircles, cv::HOUGH_GRADIENT, 1.85, binaryImg.rows / 16, 80, 30, 5, 35);

    // �������� �������� ����������� ��� ����������� ������
    detectedImg = src.clone();
    cv::cvtColor(detectedImg, detectedImg, cv::COLOR_GRAY2BGR);

    // ���������� ��������������� �����
    for (const auto& circle : detectedCircles) {
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(detectedImg, center, radius, cv::Scalar(0, 255, 0), 2);
    }

    // �������� ����������� ����������� � ���������������� �������
    cv::imshow("Detected Circles", detectedImg);

    // ������ �������� ��������
    double iouThreshold = 0.5; // ����� IoU ��� ������� �������� TP
    evaluateDetections(detectedCircles, groundTruths, iouThreshold);
}

// ������� ��� �������� ����������� �������
void createGroundTruth(cv::Mat& img, std::vector<cv::Vec3f>& groundTruths) {
    for (int y = 40; y < img.rows; y += 80) {
        for (int x = 40; x < img.cols; x += 80) {
            double radius = 5.0 + (x / 80) * 30.0 / 9.0;
            double colorValue = 97.5 + (y / 80) * (255.0 - 97.5) / 9.0;
            cv::circle(img, cv::Point(x, y), radius, cv::Scalar(colorValue), -1);
            groundTruths.push_back(cv::Vec3f(x, y, radius));
        }
    }
}

int main(int argc, char* argv[]) {
    std::string filename = "C://opencv/lab04_/build/opencvReallyPic.jpg";

    // ��������� �����������
    src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Couldn't open the image file." << std::endl;
        return 1;
    }

    // ���������� ����������� �� �������� 800x800
    cv::resize(src, src, cv::Size(800, 800));

    // ������� ����
    cv::namedWindow("Binary with Trackbars");
    cv::namedWindow("Detected Circles");

    // ������� �������
    cv::createTrackbar("Threshold", "Binary with Trackbars", &thresholdValue, 255, onTrackbar);

    // ������� � ���������� �������� ground truth
    cv::Mat groundTruthImg = src.clone();
    cv::cvtColor(groundTruthImg, groundTruthImg, cv::COLOR_GRAY2BGR);
    createGroundTruth(groundTruthImg, groundTruths);

    onTrackbar(thresholdValue, nullptr);

    cv::waitKey(0);

    return 0;
}
