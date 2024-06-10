#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <cmath>

// Глобальные переменные
cv::Mat src, binaryImg, detectedImg;
int thresholdValue = 128;
int param1Value = 80;
int param2Value = 30;
std::vector<cv::Vec3f> groundTruths;

// Функция для бинаризации изображения
void binaryThreshold(const cv::Mat& src, cv::Mat& dst, int threshold) {
    cv::threshold(src, dst, threshold, 255, cv::THRESH_BINARY);
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
void evaluateFROC(const std::vector<std::vector<cv::Vec3f>>& allDetections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) {
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

        double sensitivity = (double)TP / (TP + FN);
        double fppi = (double)FP / 100.0; // на одно изображение

        frocPoints.push_back(std::make_tuple(fppi, sensitivity));
    }

    for (const auto& point : frocPoints) {
        std::cout << "FPPI: " << std::get<0>(point) << ", Sensitivity: " << std::get<1>(point) << std::endl;
    }
}

// Обратный вызов для трекбара
void onTrackbar(int, void*) {
    // Бинаризация изображения
    binaryThreshold(src, binaryImg, thresholdValue);
    cv::imshow("Binary Image", binaryImg);
    cv::imwrite("binaryImage.png", binaryImg); // Сохраняем бинарное изображение

    // Детектирование кругов методом Хаффа
    std::vector<cv::Vec3f> detectedCircles;
    cv::HoughCircles(binaryImg, detectedCircles, cv::HOUGH_GRADIENT, 1, binaryImg.rows / 16, param1Value, param2Value, 5, 35);

    // Копируем исходное изображение для отображения кругов
    detectedImg = src.clone();
    cv::cvtColor(detectedImg, detectedImg, cv::COLOR_GRAY2BGR);

    // Отображаем детектированные круги
    for (const auto& circle : detectedCircles) {
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(detectedImg, center, radius, cv::Scalar(0, 255, 0), 2);
    }

    // Показать обновленное изображение с детектированными кругами
    cv::imshow("Detected Circles", detectedImg);
    cv::imwrite("detectedCircles.png", detectedImg); // Сохраняем изображение с детектированными кругами

    // Оценка качества детекции
    double iouThreshold = 0.5; // Порог IoU для считать детекцию TP
    evaluateDetections(detectedCircles, groundTruths, iouThreshold);

    // Оценка FROC
    static std::vector<std::vector<cv::Vec3f>> allDetections;
    allDetections.push_back(detectedCircles);
    evaluateFROC(allDetections, groundTruths, iouThreshold);
}

// Функция для разметки изображения кругами
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
    std::string filename = "C://opencv/lab04/build/LoG1.png";

    // Открываем изображение
    src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Couldn't open the image file." << std::endl;
        return 1;
    }

    // Уменьшение изображения до размеров 800x800
    cv::resize(src, src, cv::Size(800, 800));

    // Создаем окна
    cv::namedWindow("Binary with Trackbars");
    cv::namedWindow("Binary Image");
    cv::namedWindow("Detected Circles");
    cv::namedWindow("Ground Truth");

    // Создаем трекбары
    cv::createTrackbar("Threshold", "Binary with Trackbars", &thresholdValue, 255, onTrackbar);
    cv::createTrackbar("Param1", "Binary with Trackbars", &param1Value, 200, onTrackbar);
    cv::createTrackbar("Param2", "Binary with Trackbars", &param2Value, 100, onTrackbar);

    // Создаем и отображаем разметку ground truth
    cv::Mat groundTruthImg = src.clone();
    cv::cvtColor(groundTruthImg, groundTruthImg, cv::COLOR_GRAY2BGR);
    createGroundTruth(groundTruthImg, groundTruths);
    cv::imshow("Ground Truth", groundTruthImg);
    cv::imwrite("groundTruth.png", groundTruthImg); // Сохраняем ground truth

    onTrackbar(0, nullptr);

    cv::waitKey(0);

    return 0;
}
