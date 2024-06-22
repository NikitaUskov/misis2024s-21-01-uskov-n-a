#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip> // для std::setprecision
#include <sstream>
#include <string>

class AbsoluteComparison {
public:
    AbsoluteComparison(double threshold = 0.9)
        : threshold(threshold) {}

    // Функция для изменения размера изображения
    cv::Mat resizeImage(const cv::Mat& image, int width, int height) const {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height));
        return resized;
    }

    // Функция для предварительной обработки изображения: серый цвет, размывка и Canny
    cv::Mat preprocessImage(const cv::Mat& image, const std::string& baseFilename) const {
        cv::Mat gray, blurred, canny;

        // Преобразование в серый цвет
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::imwrite(baseFilename + "_gray.jpg", gray);

        // Размытие Гаусса
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::imwrite(baseFilename + "_blurred.jpg", blurred);

        // Применение фильтра Canny
        cv::Canny(blurred, canny, 50, 150);
        cv::imwrite(baseFilename + "_canny.jpg", canny);

        return canny;
    }

    // Функция для сравнения изображений по абсолютной разности
    double compareImages(const cv::Mat& img1, const cv::Mat& img2) const {
        // Проверка на совпадение размеров изображений
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            throw std::invalid_argument("Input images must have the same size and type.");
        }

        // Применяем абсолютную разницу
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);

        // Сохранение для анализа
        cv::imwrite("difference.jpg", diff);

        // Конвертация для нормализации
        diff.convertTo(diff, CV_32F);

        // Вычисление суммы абсолютных разностей
        double sum_diff = cv::sum(diff)[0];

        // Нахождение максимальной возможной разности
        double max_diff = img1.total() * 255.0;

        // Нормализация от 0 до 1 (где 1 - полное совпадение)
        double similarity = 1.0 - (sum_diff / max_diff);

        return similarity;
    }

    // Функция для получения области интереса (ROI)
    cv::Rect getROI() const {
        int x = 2400;  // начальная координата x
        int y = 2050;  // начальная координата y
        int width = 1600;  // ширина ROI
        int height = 2680;  // высота ROI

        return cv::Rect(x, y, width, height);
    }

    // Основной метод для выполнения сравнения
    void runComparison(const std::string& testImagePath, const std::string& closedImagePath) const {
        // Загружаем изображения
        cv::Mat testImage = cv::imread(testImagePath);
        cv::Mat closedImage = cv::imread(closedImagePath);

        if (testImage.empty() || closedImage.empty()) {
            std::cout << "Could not open or find the images!" << std::endl;
            return;
        }

        // Получаем ROI для обоих изображений
        cv::Rect roi = getROI();
        cv::Mat testImageROI = testImage(roi);
        cv::Mat closedImageROI = closedImage(roi);

        // Предварительная обработка изображений
        cv::Mat processedTestImageROI = preprocessImage(testImageROI, "testImageROI");
        cv::Mat processedClosedImageROI = preprocessImage(closedImageROI, "closedImageROI");

        // Сравнение обработанных изображений
        double similarity = compareImages(processedTestImageROI, processedClosedImageROI);

        // Создаем изображение с результатом
        cv::Mat resultImage = cv::Mat::zeros(200, 650, CV_8UC3);
        cv::putText(resultImage, getResultMessage(similarity), cv::Point(50, 100),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imwrite("result_image.jpg", resultImage);

        // Изменяем размер и выводим обработанные изображения для визуализации
        cv::Mat resizedProcessedTestImageROI = resizeImage(processedTestImageROI, 600, 600);
        cv::Mat resizedProcessedClosedImageROI = resizeImage(processedClosedImageROI, 600, 600);
        cv::imshow("Processed Test Window ROI", resizedProcessedTestImageROI);
        cv::imshow("Processed Closed Window ROI", resizedProcessedClosedImageROI);
        cv::imshow("Result Image", resultImage);

        // Ожидаем нажатия клавиши, чтобы закрыть окна
        cv::waitKey(0);
    }

private:
    double threshold;

    // Функция для формирования строки с сообщением о результате
    std::string getResultMessage(double similarity) const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (similarity > threshold) {
            ss << "Window is closed. Similarity: " << similarity * 100 << "%";
        }
        else {
            ss << "Window is open. Similarity: " << similarity * 100 << "%";
        }
        return ss.str();
    }
};

int main() {
    // Получаем путь к тестовому изображению из аргументов командной строки
    std::string testImagePath = "C://opencv/CW/Win_Close1.jpg";
    std::string closedImagePath = "C://opencv/CW/Win_True_Close.jpg";

    // Создаем объект класса AbsoluteComparison
    AbsoluteComparison comparator(0.95);

    // Выполняем сравнение изображений
    comparator.runComparison(testImagePath, closedImagePath);

    return 0;
}