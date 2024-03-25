#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <numeric>




int main(int argc, char* argv[]) {

    double rad, size;
    size = 99.0;
    rad = 25.0;
    std::vector<double> col = { 0, 127.0, 255.0};

    std::vector<cv::Mat> pic(6);
    cv::Mat img, imgF;

    for (int i = 0; i < pic.size(); i++) {
        cv::Mat back(size, size, CV_8UC1, cv::Scalar(col[i % 3]));
        pic[i] = back;
        if (i < 3) {
            cv::circle(pic[i], cv::Point(size / 2, size / 2), rad, cv::Scalar(col[(i + 1) % 3]), -1);
        }
        else{cv::circle(pic[i], cv::Point(size / 2, size / 2), rad, cv::Scalar(col[(i+2) % 3]), -1); }
    }

    std::vector<cv::Mat> column(3);

    for (int i = 0; i < column.size(); i++) {
        cv::vconcat(pic[i], pic[i+3], column[i]);
    }
    cv::hconcat(column, img);
    img.convertTo(imgF, CV_32FC1, 1, 0);

    cv::Mat FiltImg1(size * 2, size * 3, CV_32FC1);
    cv::Mat FiltImg2(size * 2, size * 3, CV_32FC1);
    cv::Mat FiltImg3(size * 2, size * 3, CV_32FC1);
    

    cv::Mat R1 = (cv::Mat_<float>(2, 2) << -1, 1, -1, 1);
    cv::Mat R2 = (cv::Mat_<float>(2, 2) << 1, 1, -1, -1);


    cv::filter2D(imgF, FiltImg1, -1, R1);
    cv::filter2D(imgF, FiltImg2, -1, R2);


    cv::magnitude(FiltImg1, FiltImg2, FiltImg3);

    
    cv::normalize(FiltImg1, FiltImg1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(FiltImg2, FiltImg2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(FiltImg3, FiltImg3,0,255, cv::NORM_MINMAX,CV_8UC1);

    std::vector<cv::Mat> pictures = { FiltImg1, FiltImg2, FiltImg3 };
    cv::Mat Final(size * 2, size * 3, CV_8UC3);
    cv::merge(pictures, Final);


    //cv::imshow("img", img);
    //cv::imshow("R1", FiltImg1);
    //cv::imshow("R2", FiltImg2);
    //cv::imshow("sqrt(R1+R2)", FiltImg3);
    //cv::imshow("Final", Final);
    cv::imwrite("OrigImg.png", img);
    cv::imwrite("R1.png", FiltImg1);
    cv::imwrite("R2.png", FiltImg2);
    cv::imwrite("sqrt(R1+R2).png", FiltImg3);
    cv::imwrite("Final.png", Final);
    cv::waitKey(0);

    return 0;
}
