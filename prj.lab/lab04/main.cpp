#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

// Глобальные переменные
cv::Mat src, noisyImg, binaryImg, detectedImg;
int thresholdValue = 128;
int blurSize = 5;
double noiseStdDev = 25.0;  // Стандартное отклонение для шума
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

    // Добавляем текст с параметрами на изображение
    std::string tpText = "TP: " + std::to_string(TP);
    std::string fpText = "FP: " + std::to_string(FP);
    std::string fnText = "FN: " + std::to_string(FN);
    cv::putText(detectedImg, tpText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(detectedImg, fpText, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(detectedImg, fnText, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
}

// Функция для детекции окружностей и оценка детекций
void detectAndEvaluateCircles(const cv::Mat& binaryImg, const cv::Mat& src, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) {
    std::vector<cv::Vec3f> detectedCircles;
    int maxRad = 800 / (4 * std::fmax(binaryImg.rows, binaryImg.cols));
    cv::HoughCircles(binaryImg, detectedCircles, cv::HOUGH_GRADIENT, 1.8, binaryImg.rows / 16, 80, 30, 5, 40);

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
}

// Обратный вызов для трекбаров
void onTrackbar(int, void*) {
    // Выбор метода бинаризации
    switch (binarizationMethod) {
    case 0:
        binaryThreshold(noisyImg, binaryImg, thresholdValue, blurSize);
        break;
    case 1:
        adaptiveThresholding(noisyImg, binaryImg, blurSize * 2 + 1, 5);
        break;
    case 2:
        otsuThresholding(noisyImg, binaryImg);
        break;
    }

    // Показать исходное изображение
    cv::imshow("Original Image", src);

    // Показать изображение с шумом
    cv::imshow("Noisy Image", noisyImg);

    // Показать обновленное бинаризированное изображение
    cv::imshow("Binary with Trackbars", binaryImg);

    // Детектирование кругов и оценка качества
    double iouThreshold = 0.5; // Порог IoU для считать детекцию TP
    detectAndEvaluateCircles(binaryImg, noisyImg, groundTruths, iouThreshold);
}

// Обработчик событий мыши для сохранения изображения
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        saveImage = true;
        std::string outputFilename = "detected_image.jpg";
        cv::imwrite(outputFilename, detectedImg);
        std::cout << "Image saved as " << outputFilename << std::endl;
    }
}

// Функция для создания изображения и разметки ground truth
void createGroundTruth(cv::Mat& img, std::vector<cv::Vec3f>& groundTruths) {
    int rows = img.rows / 160;
    int cols = img.cols / 80;

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

int main(int argc, char* argv[]) {
    // Создаем пустое изображение
    src = cv::Mat::zeros(800, 800, CV_8UC1);

    // Создаем и отображаем разметку ground truth
    createGroundTruth(src, groundTruths);

    // Добавляем шум к изображению
    addNoise(src, noisyImg, noiseStdDev);

    // Создаем окна
    cv::namedWindow("Original Image");
    cv::namedWindow("Noisy Image");
    cv::namedWindow("Binary with Trackbars");
    cv::namedWindow("Detected Circles");

    // Создаем трекбары
    cv::createTrackbar("Threshold", "Binary with Trackbars", &thresholdValue, 255, onTrackbar);
    cv::createTrackbar("Blur Size", "Binary with Trackbars", &blurSize, 50, onTrackbar);
    cv::createTrackbar("Method: 0-Simple, 1-Adaptive, 2-Otsu", "Binary with Trackbars", &binarizationMethod, 2, onTrackbar);

    // Назначаем обработчик событий мыши
    cv::setMouseCallback("Detected Circles", onMouse);

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
