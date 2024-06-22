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

    // Функция для сравнения изображений по абсолютной разности
    double compareImages(const cv::Mat& img1, const cv::Mat& img2) const {
        // Проверка на совпадение размеров изображений
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            throw std::invalid_argument("Input images must have the same size and type.");
        }

        // Конвертация в серый цвет
        cv::Mat grayImg1, grayImg2;
        cv::cvtColor(img1, grayImg1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, grayImg2, cv::COLOR_BGR2GRAY);

        // Размывка
        cv::Mat blurredImg1, blurredImg2;
        cv::GaussianBlur(grayImg1, blurredImg1, cv::Size(5, 5), 1);
        cv::GaussianBlur(grayImg2, blurredImg2, cv::Size(5, 5), 1);

        // Вычисление абсолютной разницы между изображениями
        cv::Mat diff;
        cv::absdiff(blurredImg1, blurredImg2, diff);

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
        // Определите координаты и размеры окна (ROI) здесь.
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

        // Изменяем размер обрезанных изображений для вывода
        cv::Mat resizedTestImageROI = resizeImage(testImageROI, 600, 600);
        cv::Mat resizedClosedImageROI = resizeImage(closedImageROI, 600, 600);

        // Промежуточная обработка и сравнение изображений
        double similarity = compareImages(testImageROI, closedImageROI);

        // Создаем изображение с результатом
        cv::Mat resultImage = cv::Mat::zeros(200, 1050, CV_8UC3);
        cv::putText(resultImage, getResultMessage(similarity), cv::Point(50, 100),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Выводим изображение с результатом
        cv::imshow("Result Image - AbsoluteComparison", resultImage);
        cv::waitKey(0);

        // Сохраняем изображение с результатом на диск
        cv::imwrite("absolute_result_image.jpg", resultImage);

        // Определяем порог для различия
        if (similarity > threshold) {
            std::cout << "The window is closed. Method: absdiff, Similarity: " << std::fixed << std::setprecision(2) << similarity * 100 << "%" << std::endl;
        }
        else {
            std::cout << "The window is open. Method: absdiff, Similarity: " << std::fixed << std::setprecision(2) << similarity * 100 << "%" << std::endl;
        }
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

class SIFTComparison : public AbsoluteComparison {
public:
    SIFTComparison() {}

    void runComparison(const std::string& testImagePath, const std::string& closedImagePath) const {
        cv::Mat testImage = cv::imread(testImagePath);
        cv::Mat closedImage = cv::imread(closedImagePath);

        if (testImage.empty() || closedImage.empty()) {
            std::cout << "Could not open or find the images!" << std::endl;
            return;
        }

        cv::Rect roi = getROI();
        cv::Mat testImageROI = testImage(roi);
        cv::Mat closedImageROI = closedImage(roi);

        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(testImageROI, cv::noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(closedImageROI, cv::noArray(), keypoints2, descriptors2);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        // Применение RANSAC для отсеивания ложных срабатываний
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

        std::vector<uchar> inliersMask;
        cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, inliersMask);

        std::vector<cv::DMatch> inliers;
        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i]) {
                inliers.push_back(matches[i]);
            }
        }

        double avgDisplacement = calculateAverageDisplacement(keypoints1, keypoints2, inliers);

        // Создаем изображение с результатом для SIFTComparison
        cv::Mat resultImage = cv::Mat::zeros(200, 1050, CV_8UC3);
        cv::putText(resultImage, getResultMessage(avgDisplacement), cv::Point(50, 100),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Выводим изображение с результатом для SIFTComparison
        cv::imshow("Result Image - SIFTComparison", resultImage);
        cv::waitKey(0);

        // Сохраняем изображение с результатом на диск
        cv::imwrite("sift_result_image.jpg", resultImage);

        // Выводим изображение с найденными ключевыми точками и совпадениями
        cv::Mat imgMatches;
        cv::drawMatches(testImageROI, keypoints1, closedImageROI, keypoints2, inliers, imgMatches);

        // Выводим изображение с найденными ключевыми точками и совпадениями
        cv::imshow("SIFT Matches", imgMatches);
        cv::waitKey(0);

        // Сохраняем изображение с совпадениями на диск
        cv::imwrite("sift_matches.jpg", imgMatches);

        // Ожидаем нажатия клавиши, чтобы закрыть окна
        cv::waitKey(0);
    }

private:
    double calculateAverageDisplacement(const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::DMatch>& matches) const {
        double totalDisplacement = 0.0;
        for (const auto& match : matches) {
            cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
            double displacement = cv::norm(pt1 - pt2);
            totalDisplacement += displacement;
        }
        return totalDisplacement / matches.size();
    }

    std::string getResultMessage(double avgDisplacement) const {
        const double threshold = 10.0;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (avgDisplacement < threshold) {
            ss << "Window is closed. Average displacement: " << avgDisplacement << " pixels.";
        }
        else {
            ss << "Window is open. Average displacement: " << avgDisplacement << " pixels.";
        }
        return ss.str();
    }
};

int main() {
    // Получаем путь к тестовому изображению из аргументов командной строки
    std::string testImagePath = "C://opencv/CW/Win_Open2.jpg";
    std::string closedImagePath = "C://opencv/CW/Win_True_Close.jpg";

    // Создаем объекты сравнения
    AbsoluteComparison absoluteComparator(0.95);
    SIFTComparison siftComparator;

    // Выполняем сравнение и выводим результаты
    absoluteComparator.runComparison(testImagePath, closedImagePath);
    siftComparator.runComparison(testImagePath, closedImagePath);

    return 0;
}
