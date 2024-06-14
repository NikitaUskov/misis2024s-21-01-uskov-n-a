#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Загружаем изображение
    cv::Mat image = cv::imread("C://Users/79194/OneDrive - НИТУ МИСиС/Рисунки/Рандом фотки/Window.jpg");
    if (image.empty()) {
        std::cerr << "Ошибка: Не удалось загрузить изображение." << std::endl;
        return -1;
    }

    // Уменьшаем размер изображения до 800x800 пикселей
    cv::Size targetSize(800, 800);
    cv::resize(image, image, targetSize);

    // Преобразуем изображение в оттенки серого
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Применяем размытие для уменьшения шума
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 2);

    cv::imshow("Gauss", gray);

    // Применяем метод Canny для нахождения границ
    cv::Mat edges;
    cv::Canny(gray, edges, 25, 180);

    cv::imshow("Canny", edges);

    // Находим контуры
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Нарисовать прямоугольники вокруг контуров
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        if (cv::contourArea(contour) > 100) { // Игнорируем маленькие контуры
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Показываем изображение с найденными прямоугольниками
    cv::imshow("Распознование прямоугольников", image);
    cv::waitKey(0);

    return 0;
}
