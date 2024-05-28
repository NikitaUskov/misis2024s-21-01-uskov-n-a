#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Импорт изображения
    cv::Mat img = cv::imread("C:/Users/79194/OneDrive - НИТУ МИСиС/Рисунки/Рандом фотки/Собака.jpg", cv::IMREAD_COLOR);

    // Проверка, что изображение успешно загружено
    if (img.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // Показать оригинальное изображение
    cv::imshow("GrayWorld_before", img);

    // Определение каналов BGR
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(img, bgrChannels);
    cv::Mat& b = bgrChannels[0];
    cv::Mat& g = bgrChannels[1];
    cv::Mat& r = bgrChannels[2];

    // Вычисление среднего значения для каналов BGR
    double averB = cv::mean(b)[0];
    double averG = cv::mean(g)[0];
    double averR = cv::mean(r)[0];

    // Вычисление среднего серого значения
    double grayValue = (averR + averB + averG) / 3.0;

    // Вычисление коэффициентов усиления
    double kb = grayValue / averB;
    double kg = grayValue / averG;
    double kr = grayValue / averR;

    // Усиление каналов
    b *= kb;
    g *= kg;
    r *= kr;

    // Слияние каналов обратно в изображение
    cv::merge(bgrChannels, img);

    // Показать обработанное изображение
    cv::imshow("GrayWorld_after", img);

    // Обработка событий клавиатуры
    while (true) {
        int k = cv::waitKey(1) & 0xFF;
        // Если нажата клавиша ESC, завершить отображение
        if (k == 27) {
            break;
        }
    }

    // Закрыть все окна
    cv::destroyAllWindows();
    return 0;
}
