#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

static void GammaCorrect(const cv::Mat& invec, double g, cv::Mat& outvec) {
    cv::Mat invec_float;
    invec.convertTo(invec_float, CV_32F);  // Преобразование в тип float
    cv::pow(invec_float / 255.0, g, outvec);
    outvec *= 255.0;
    outvec.convertTo(outvec, CV_8UC1);
}

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv,
        "{s|3|width}"
        "{h|20|height}"
        "{gamma|2.4|gamma}"
        "{@img||}"
    );

    int s = parser.get<int>("s");
    int h = parser.get<int>("h");
    double gamma = parser.get<double>("gamma");
    std::string filename = parser.get<std::string>("@img");

    // Создание изображения с одним каналом (CV_8UC1)
    cv::Mat orig(h, 255 * s, CV_8UC1, cv::Scalar(255));
    cv::Mat outvec(h, 255 * s, CV_8UC1, cv::Scalar(255));

    // Градиентная заливка
    for (int i = 0; i < 255; i++) {
        cv::Rect rec(i * s, 0, s, h);
        orig(rec).setTo(i);
    }

    // Гамма-коррекция
    GammaCorrect(orig, gamma, outvec);

    // Горизонтальное объединение изображений
    cv::Mat result;
    cv::vconcat(orig, outvec, result);

    // Отображение изображения
    if (filename.empty()) {
        cv::imshow("Result", result);
        // Ожидание нажатия клавиши
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else cv::imwrite(filename, result);

    return 0;
}