#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <cmath>

// Глобальные переменные
cv::Mat src, noisyImg, binaryImg, detectedImg;
int thresholdValue = 128;
int blurSize = 5;
int param1Value = 80;
int param2Value = 30;
double noiseStdDev = 40.0;  // Стандартное отклонение для шума
std::vector<cv::Vec3f> groundTruths;
bool saveImage = false;
int binarizationMethod = 0; // 0: Simple, 1: Adaptive, 2: Otsu

// Функция для добавления шума в изображение
void addNoise(const cv::Mat& src, cv::Mat& dst, double stddev) {
    cv::Mat noise = cv::Mat(src.size(), CV_64F);
    cv::randn(noise, 0, stddev);
    src.convertTo(dst, CV_64F);
    dst += noise;
    dst.convertTo(dst, CV_8U);
}

// Функция для простой бинаризации изображения
void binaryThreshold(const cv::Mat& src, cv::Mat& dst, int threshold, int blur) {
    cv::GaussianBlur(src, dst, cv::Size(blur * 2 + 1, blur * 2 + 1), 0); // Size must be odd
    cv::threshold(dst, dst, threshold, 255, cv::THRESH_BINARY);
}

// Функция для адаптивной бинаризации
void adaptiveThresholding(const cv::Mat& src, cv::Mat& dst, int blockSize, double C) {
    cv::adaptiveThreshold(src, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
}

// Функция для бинаризации по методу Оцу
void otsuThresholding(const cv::Mat& src, cv::Mat& dst) {
    cv::threshold(src, dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

// Функция для оценки детекций по IoU
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
            double iou = (distance <= radiusSum) ? 1.0 : 0.0; // Упростим вычисление IoU

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

// Функция для оценки FROC
std::vector<std::tuple<double, double>> evaluateFROC(const std::vector<std::vector<cv::Vec3f>>& allDetections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) {
    std::vector<std::tuple<double, double>> frocPoints;

    for (const auto& detections : allDetections) {
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
                double iou = (distance <= radiusSum) ? 1.0 : 0.0;

                if (iou >= iouThreshold) {
                    TP++;
                    detected = true;
                    break;
                }
            }
            if (!detected) FN++;
        }

        FP = detections.size() - TP;

        double sensitivity = static_cast<double>(TP) / (TP + FN);
        double fppi = static_cast<double>(FP) / 1.0; // на одно изображение

        frocPoints.push_back(std::make_tuple(fppi, sensitivity));
    }

    for (const auto& point : frocPoints) {
        std::cout << "FPPI: " << std::get<0>(point) << ", Sensitivity: " << std::get<1>(point) << std::endl;
    }

    return frocPoints;
}

// Функция для отображения FROC кривой
void drawFROC(const std::vector<std::tuple<double, double>>& frocPoints, cv::Mat& frocImg) {
    // Определим размеры изображения и отступы
    int width = 600, height = 400;
    int margin = 50;
    frocImg = cv::Mat::zeros(height, width, CV_8UC3);

    // Нарисуем оси
    cv::line(frocImg, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(255, 255, 255), 2);
    cv::line(frocImg, cv::Point(margin, height - margin), cv::Point(margin, margin), cv::Scalar(255, 255, 255), 2);

    // Подписи осей
    cv::putText(frocImg, "FPPI", cv::Point(width / 2, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(frocImg, "Sensitivity", cv::Point(10, height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Определим масштабирование
    double maxFPPI = 100;
    double maxSensitivity = 1.0;

    // Нарисуем точки FROC
    for (const auto& point : frocPoints) {
        double fppi = std::get<0>(point);
        double sensitivity = std::get<1>(point);

        int x = margin + static_cast<int>((width - 2 * margin) * (fppi / maxFPPI));
        int y = height - margin - static_cast<int>((height - 2 * margin) * (sensitivity / maxSensitivity));

        cv::circle(frocImg, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("FROC Curve", frocImg);
    cv::imwrite("frocCurve.png", frocImg); // Сохраняем FROC кривую
}

// Функция для детекции окружностей и оценка детекций
void detectAndEvaluateCircles(const cv::Mat& binaryImg, const cv::Mat& src, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) {
    std::vector<cv::Vec3f> detectedCircles;
    cv::HoughCircles(binaryImg, detectedCircles, cv::HOUGH_GRADIENT, 1, binaryImg.rows / 16, param1Value, param2Value, 2, 35);

    // Копируем исходное изображение для отображения кругов
    detectedImg = src.clone();
    cv::cvtColor(detectedImg, detectedImg, cv::COLOR_GRAY2BGR);

    // Отображаем детектированные круги
    for (const auto& circle : detectedCircles) {
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(detectedImg, center, radius, cv::Scalar(0, 255, 0), 2);
    }

    // Оценка качества детекции
    evaluateDetections(detectedCircles, groundTruths, iouThreshold);

    // Показать обновленное изображение с детектированными кругами и параметрами
    cv::imshow("Detected Circles", detectedImg);
    cv::imwrite("detectedCircles.png", detectedImg); // Сохраняем изображение с детектированными кругами

    // Оценка FROC
    static std::vector<std::vector<cv::Vec3f>> allDetections;
    allDetections.push_back(detectedCircles);
    auto frocPoints = evaluateFROC(allDetections, groundTruths, iouThreshold);

    // Отображение FROC кривой
    cv::Mat frocImg;
    drawFROC(frocPoints, frocImg);
}

// Функция для создания ground truth кругов на изображении
void createGroundTruth(cv::Mat& img, std::vector<cv::Vec3f>& groundTruths) {
    int rows = img.rows / 50;
    int cols = img.cols / 50;

    // Максимальное число окружностей, которое может быть
    int maxCircles = rows * cols;

    // Максимальные значения для яркости и радиуса
    double maxColorValue = 250.0;
    double maxRadius = 800 / (2 * std::fmax(rows, cols));

    // Вычисляем шаг для яркости и радиуса
    double colorStep = maxColorValue / (rows - 1);
    double radiusStep = maxRadius / (cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int x = img.cols / (2 * cols) + j * (img.cols / cols);
            int y = img.rows / (2 * rows) + i * (img.rows / rows);

            // Вычисляем яркость и радиус на основании текущего индекса
            double radius = radiusStep + j * radiusStep;
            double colorValue = 80.0 + i * colorStep;

            // Ограничиваем максимальные значения
            if (groundTruths.size() >= maxCircles) {
                radius = maxRadius;
                colorValue = maxColorValue;
            }

            cv::circle(img, cv::Point(x, y), radius, cv::Scalar(colorValue), -1);
            groundTruths.push_back(cv::Vec3f(x, y, radius));
        }
    }
}

// Обратный вызов для трекбара
void onTrackbar(int, void*) {
    // Бинаризация изображения с учетом метода бинаризации
    if (binarizationMethod == 0) {
        binaryThreshold(noisyImg, binaryImg, thresholdValue, blurSize);
    }
    else if (binarizationMethod == 1) {
        adaptiveThresholding(noisyImg, binaryImg, blurSize * 2 + 1, 5);
    }
    else if (binarizationMethod == 2) {
        otsuThresholding(noisyImg, binaryImg);
    }

    cv::imshow("Binary with Trackbars", binaryImg);
    cv::imwrite("binaryImage.png", binaryImg); // Сохраняем бинарное изображение

    // Детектирование кругов методом Хаффа и оценка результатов
    detectAndEvaluateCircles(binaryImg, src, groundTruths, 0.5);
}

int main(int argc, char* argv[]) {
    // Создаем пустое изображение
    src = cv::Mat::zeros(800, 800, CV_8UC1);

    // Создаем и отображаем разметку ground truth
    createGroundTruth(src, groundTruths);
    cv::imshow("Ground Truth", src);
    cv::imwrite("groundTruth.png", src); // Сохраняем ground truth

    // Добавляем шум к изображению
    addNoise(src, noisyImg, noiseStdDev);
    cv::imshow("Noisy Image", noisyImg);
    cv::imwrite("Noisy_Image.jpg", noisyImg); // Сохраняем изображение с шумом

    // Создаем окна
    cv::namedWindow("Binary with Trackbars");
    cv::namedWindow("Detected Circles");

    // Создаем трекбары
    cv::createTrackbar("Threshold", "Binary with Trackbars", &thresholdValue, 255, onTrackbar);
    cv::createTrackbar("Blur Size", "Binary with Trackbars", &blurSize, 50, onTrackbar);
    cv::createTrackbar("Param1", "Binary with Trackbars", &param1Value, 200, onTrackbar);
    cv::createTrackbar("Param2", "Binary with Trackbars", &param2Value, 100, onTrackbar);
    cv::createTrackbar("Method: 0-Simple, 1-Adaptive, 2-Otsu", "Binary with Trackbars", &binarizationMethod, 2, onTrackbar);

    // Инициализируем бинаризированное изображение и запустим обработку
    binaryImg = noisyImg.clone();
    onTrackbar(thresholdValue, nullptr);

    while (true) {
        int key = cv::waitKey(0);
        if (key == 27) {  // ESC key
            break;
        }
        else if (key == 's' || key == 'S') {
            std::string outputFilename = "detected_image.jpg";
            cv::imwrite(outputFilename, detectedImg);
            cv::imwrite("Original_Image.jpg", src);
            cv::imwrite("Noisy_Image.jpg", noisyImg);
            cv::imwrite("Binary_with_Trackbars.jpg", binaryImg);
            std::cout << "Image saved as " << outputFilename << std::endl;
        }
    }

    return 0;
}
