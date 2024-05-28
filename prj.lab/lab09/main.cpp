#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // ������ �����������
    cv::Mat img = cv::imread("C:/Users/79194/OneDrive - ���� �����/�������/������ �����/������.jpg", cv::IMREAD_COLOR);

    // ��������, ��� ����������� ������� ���������
    if (img.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // �������� ������������ �����������
    cv::imshow("GrayWorld_before", img);

    // ����������� ������� BGR
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(img, bgrChannels);
    cv::Mat& b = bgrChannels[0];
    cv::Mat& g = bgrChannels[1];
    cv::Mat& r = bgrChannels[2];

    // ���������� �������� �������� ��� ������� BGR
    double averB = cv::mean(b)[0];
    double averG = cv::mean(g)[0];
    double averR = cv::mean(r)[0];

    // ���������� �������� ������ ��������
    double grayValue = (averR + averB + averG) / 3.0;

    // ���������� ������������� ��������
    double kb = grayValue / averB;
    double kg = grayValue / averG;
    double kr = grayValue / averR;

    // �������� �������
    b *= kb;
    g *= kg;
    r *= kr;

    // ������� ������� ������� � �����������
    cv::merge(bgrChannels, img);

    // �������� ������������ �����������
    cv::imshow("GrayWorld_after", img);

    // ��������� ������� ����������
    while (true) {
        int k = cv::waitKey(1) & 0xFF;
        // ���� ������ ������� ESC, ��������� �����������
        if (k == 27) {
            break;
        }
    }

    // ������� ��� ����
    cv::destroyAllWindows();
    return 0;
}
