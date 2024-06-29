#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip> // для std::setprecision
#include <sstream>
#include <string>
#include <vector>
#include <filesystem> 
#include <numeric>
#include <algorithm>

namespace fs = std::filesystem;

// Функция для рисования текста по центру ячейки
void drawCenteredText(cv::Mat& img, const std::string& text, cv::Point& pt, int width, int height) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    baseline += 1;

    // Координаты текста для центрирования
    cv::Point textOrg((width - textSize.width) / 2 + pt.x, (height + textSize.height) / 2 + pt.y);

    cv::putText(img, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// Функция для рисования таблицы
void drawTable(cv::Mat& img, const std::vector<int>& imageNumbers, const std::vector<std::string>& statesAbsDiff,
    const std::vector<std::string>& metricsABSDiff, const std::vector<std::string>& statesSIFT,
    const std::vector<std::string>& metricsSIFT) {
    // Координаты и размеры таблицы
    cv::Point origin(50, 50);
    int cellWidth = 150;
    int cellHeight = 40;

    // Названия столбцов
    std::vector<std::string> headers = { "Pic Number", "State of Abs", "Metric of Abs", "State of SIFT", "Metric of SIFT" };

    // Рисуем заголовки
    for (size_t i = 0; i < headers.size(); ++i) {
        cv::Point pt(origin.x + i * cellWidth, origin.y);
        cv::rectangle(img, pt, cv::Point(pt.x + cellWidth, pt.y + cellHeight), cv::Scalar(0, 0, 0), 1);
        drawCenteredText(img, headers[i], pt, cellWidth, cellHeight);
    }

    // Рисуем строки данных
    for (size_t i = 0; i < imageNumbers.size(); ++i) {
        // Номер картинки
        cv::Point pt1(origin.x, origin.y + (i + 1) * cellHeight);
        cv::rectangle(img, pt1, cv::Point(pt1.x + cellWidth, pt1.y + cellHeight), cv::Scalar(0, 0, 0), 1);
        drawCenteredText(img, std::to_string(imageNumbers[i]), pt1, cellWidth, cellHeight);

        // Состояние AbsDiff
        cv::Point pt2(origin.x + cellWidth, origin.y + (i + 1) * cellHeight);
        cv::rectangle(img, pt2, cv::Point(pt2.x + cellWidth, pt2.y + cellHeight), cv::Scalar(0, 0, 0), 1);
        drawCenteredText(img, statesAbsDiff[i], pt2, cellWidth, cellHeight);

        // Метрика ABSDiff
        cv::Point pt3(origin.x + 2 * cellWidth, origin.y + (i + 1) * cellHeight);
        cv::rectangle(img, pt3, cv::Point(pt3.x + cellWidth, pt3.y + cellHeight), cv::Scalar(0, 0, 0), 1);
        drawCenteredText(img, metricsABSDiff[i], pt3, cellWidth, cellHeight);

        // Состояние SIFT
        cv::Point pt4(origin.x + 3 * cellWidth, origin.y + (i + 1) * cellHeight);
        cv::rectangle(img, pt4, cv::Point(pt4.x + cellWidth, pt4.y + cellHeight), cv::Scalar(0, 0, 0), 1);
        drawCenteredText(img, statesSIFT[i], pt4, cellWidth, cellHeight);

        // Метрика SIFT
        cv::Point pt5(origin.x + 4 * cellWidth, origin.y + (i + 1) * cellHeight);
        cv::rectangle(img, pt5, cv::Point(pt5.x + cellWidth, pt5.y + cellHeight), cv::Scalar(0, 0, 0), 1);
        drawCenteredText(img, metricsSIFT[i], pt5, cellWidth, cellHeight);
    }
}

// Функция для подсчета количества 0 и 1 в булевом векторе
std::pair<int, int> countZeroesAndOnes(const std::vector<bool>& vec) {
    int count_zeroes = 0;
    int count_ones = 0;

    for (bool value : vec) {
        if (value) {
            count_ones++;
        }
        else {
            count_zeroes++;
        }
    }

    return { count_zeroes, count_ones };
}

// Функция для построения гистограммы
cv::Mat createHistogram(const std::vector<bool>& vec) {
    auto [count_zeroes, count_ones] = countZeroesAndOnes(vec);

    // Задаем параметры гистограммы
    int bar_width = 50;
    int height = 300;  // Высота гистограммы
    int padding = 50;  // Отступ между столбцами и краями изображения
    int text_padding = 10; // Отступ для текста

    // Ширина изображения для гистограммы
    int width = 2 * bar_width + 3 * padding;

    cv::Mat histogram = cv::Mat::zeros(height, width, CV_8UC3);

    // Определяем высоты столбцов
    int max_count = std::max(count_zeroes, count_ones);
    int height_zeroes = static_cast<int>((static_cast<double>(count_zeroes) / max_count) * (height - 2 * padding));
    int height_ones = static_cast<int>((static_cast<double>(count_ones) / max_count) * (height - 2 * padding));

    // Рисуем столбец для нулей
    cv::rectangle(histogram, cv::Point(padding, height - padding - height_zeroes), cv::Point(padding + bar_width, height - padding), cv::Scalar(255, 0, 0), cv::FILLED);
    // Рисуем количество нулей над столбцом
    cv::putText(histogram, std::to_string(count_zeroes), cv::Point(padding + bar_width / 2 - 10, height - padding - height_zeroes - text_padding), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // Рисуем столбец для единиц
    cv::rectangle(histogram, cv::Point(2 * padding + bar_width, height - padding - height_ones), cv::Point(2 * padding + 2 * bar_width, height - padding), cv::Scalar(0, 255, 0), cv::FILLED);
    // Рисуем количество единиц над столбцом
    cv::putText(histogram, std::to_string(count_ones), cv::Point(2 * padding + 1.5 * bar_width - 10, height - padding - height_ones - text_padding), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // Рисуем оси
    cv::line(histogram, cv::Point(padding / 2, padding / 2), cv::Point(padding / 2, height - padding), cv::Scalar(255, 255, 255), 2);
    cv::line(histogram, cv::Point(padding / 2, height - padding), cv::Point(width - padding / 2, height - padding), cv::Scalar(255, 255, 255), 2);

    // Подписи осей
    cv::putText(histogram, "closed", cv::Point(padding + bar_width / 2 - 30, height - padding + text_padding + 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    cv::putText(histogram, "open", cv::Point(2 * padding + 1.5 * bar_width - 30, height - padding + text_padding + 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    cv::putText(histogram, "Count", cv::Point(padding / 4, padding / 2), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    return histogram;
}

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
        cv::GaussianBlur(grayImg1, blurredImg1, cv::Size(5, 5), 1.5);
        cv::GaussianBlur(grayImg2, blurredImg2, cv::Size(5, 5), 1.5);

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

        cv::imwrite("absolute_diffImage.jpg", diff);

        return similarity;
    }

    // Функция для получения области интереса (ROI)
    cv::Rect getROI() const {
        // Определите координаты и размеры окна (ROI)
        int x = 1300;  // начальная координата x
        int y = 1420;  // начальная координата y
        int width = 1200;  // ширина ROI
        int height = 2100;  // высота ROI

        return cv::Rect(x, y, width, height);
    }

    // Основной метод для выполнения сравнения
    std::pair<bool, double> runABSDIFF(cv::Mat& TestImage, const std::string& closedImagePath) const {
        // Загружаем изображения
        cv::Mat testImage = TestImage;
        cv::Mat closedImage = cv::imread(closedImagePath);

        if (testImage.empty() || closedImage.empty()) {
            std::cout << "Could not open or find the images!" << std::endl;
            return { 0,-1 };
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

        // Записываем промежуточные изображения для AbsoluteComparison
        cv::imwrite("absolute_testImageROI.jpg", testImageROI);
        cv::imwrite("absolute_closedImageROI.jpg", closedImageROI);
        cv::Mat grayTestImageROI, grayClosedImageROI;
        cv::cvtColor(testImageROI, grayTestImageROI, cv::COLOR_BGR2GRAY);
        cv::cvtColor(closedImageROI, grayClosedImageROI, cv::COLOR_BGR2GRAY);
        cv::imwrite("absolute_grayTestImageROI.jpg", grayTestImageROI);
        cv::imwrite("absolute_grayClosedImageROI.jpg", grayClosedImageROI);
        cv::imwrite("absolute_resultImage.jpg", resultImage);

        // Выводим изображение с результатом
        //cv::imshow("Result Image - AbsoluteComparison", resultImage);

        std::vector<bool> test;

        // Определяем порог для различия
        if (similarity > threshold) {
            //std::cout << "The window is closed. Method: absdiff, Similarity: " << std::fixed << std::setprecision(2) << similarity * 100 << "%" << std::endl;
            return { 0, similarity };
        }
        else {
            //std::cout << "The window is open. Method: absdiff, Similarity: " << std::fixed << std::setprecision(2) << similarity * 100 << "%" << std::endl;
            return { 1, similarity };
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

    std::pair<bool, double> runSIFT(cv::Mat& TestImage, const std::string& closedImagePath) const {
        cv::Mat testImage = TestImage;
        cv::Mat closedImage = cv::imread(closedImagePath);

        if (testImage.empty() || closedImage.empty()) {
            std::cout << "Could not open or find the images!" << std::endl;
            return { 0, -1 };
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

        std::vector<double> Statistic;
        double avgDisplacement = calculateAverageDisplacement(keypoints1, keypoints2, inliers);

        Statistic.push_back(avgDisplacement);

        std::vector<bool> WindowStatus;

        // Создаем изображение с результатом для SIFTComparison
        cv::Mat resultImage = cv::Mat::zeros(200, 1050, CV_8UC3);
        cv::putText(resultImage, getResultMessage(avgDisplacement, WindowStatus), cv::Point(50, 100),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Записываем промежуточные изображения для SIFTComparison
        cv::imwrite("sift_testImageROI.jpg", testImageROI);
        cv::imwrite("sift_closedImageROI.jpg", closedImageROI);
        cv::imwrite("sift_resultImage.jpg", resultImage);

        // Выводим изображение с результатом для SIFTComparison
        //cv::imshow("Result Image - SIFTComparison", resultImage);

        // Выводим изображение с найденными ключевыми точками и совпадениями
        cv::Mat imgMatches;
        cv::drawMatches(testImageROI, keypoints1, closedImageROI, keypoints2, inliers, imgMatches);

        // Записываем изображение с найденными ключевыми точками и совпадениями
        cv::imwrite("sift_matches.jpg", imgMatches);

        // Выводим изображение с найденными ключевыми точками и совпадениями
        //cv::imshow("SIFT Matches", imgMatches);

        cv::waitKey(0);

        return { WindowStatus.back(), Statistic.back() };
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

    std::string getResultMessage(double avgDisplacement, std::vector<bool>& WindowStatus) const {
        const double threshold = 10.0;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (avgDisplacement < threshold) {
            ss << "Window is closed. Average displacement: " << avgDisplacement << " pixels.";
            WindowStatus.push_back(0);
        }
        else {
            ss << "Window is open. Average displacement: " << avgDisplacement << " pixels.";
            WindowStatus.push_back(1);
        }
        return ss.str();
    }
};


std::vector<cv::Mat> readImagesFromFolder(const std::string& folder_path) {
    std::vector<cv::Mat> images;

    try {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                cv::Mat img = cv::imread(file_path, cv::IMREAD_COLOR);  // Считывание изображения в цвете

                if (img.empty()) {
                    std::cerr << "Could not read the image: " << file_path << std::endl;
                    continue;
                }

                images.push_back(img);
            }
        }
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    return images;
}

int main() {
    // Получаем путь к тестовому изображению из аргументов командной строки
    std::string closedImagePath = "C://opencv/misis2024s-21-01-uskov-n-a/prj.cw/cw/Test_Image/True_Close.JPG";

    std::string folder_path = "C://opencv/misis2024s-21-01-uskov-n-a/prj.cw/cw/Test_Image";
    std::vector<cv::Mat> images = readImagesFromFolder(folder_path);

    // Создаем объекты сравнения
    AbsoluteComparison absoluteComparator(0.95);
    SIFTComparison siftComparator;

    std::vector<int> imageNumbers;
    std::vector<std::string> statesAbs;
    std::vector<std::string> metricsABSDiff;
    std::vector<std::string> statesSIFT;
    std::vector<std::string> metricsSIFT;

    std::vector<bool> WindowStatus_AbsDiff;
    std::vector<double> Stat_AbsDiff;

    std::vector<bool> WindowStatus_SIFT;
    std::vector<double> Stat_SIFT;

    for (size_t i = 0; i < images.size(); ++i) {

        imageNumbers.push_back(i+1);

        //AbsDiff
        WindowStatus_AbsDiff.push_back(absoluteComparator.runABSDIFF(images[i], closedImagePath).first);
        Stat_AbsDiff.push_back(absoluteComparator.runABSDIFF(images[i], closedImagePath).second);
        metricsABSDiff.push_back(std::to_string(Stat_AbsDiff.back()));
        statesAbs.push_back(WindowStatus_AbsDiff.back() ? "open" : "close");

        // SIFT
        WindowStatus_SIFT.push_back(siftComparator.runSIFT(images[i], closedImagePath).first);
        Stat_SIFT.push_back(siftComparator.runSIFT(images[i], closedImagePath).second);
        metricsSIFT.push_back(std::to_string(Stat_SIFT.back()));
        statesSIFT.push_back(WindowStatus_SIFT.back() ? "open" : "close");
    }

    // Создаем гистограмму
    cv::Mat histogram_abs = createHistogram(WindowStatus_AbsDiff);
    cv::Mat histogram_sift = createHistogram(WindowStatus_SIFT);

    double avgABS = std::accumulate(Stat_AbsDiff.begin(), Stat_AbsDiff.end(), 0);
    avgABS /= Stat_AbsDiff.size();

    double avgSIFT = std::accumulate(Stat_SIFT.begin(), Stat_SIFT.end(), 0);
    avgSIFT /= Stat_SIFT.size();


    // Создаем пустое изображение белого цвета
    cv::Mat img = cv::Mat::ones(50 + 50*WindowStatus_AbsDiff.size(), 850, CV_8UC3);
    img = cv::Scalar(255, 255, 255);

    // Рисуем таблицу
    drawTable(img, imageNumbers, statesAbs, metricsABSDiff, statesSIFT, metricsSIFT);

    // Показываем таблицу
    cv::imwrite("Table.png", img);

    // Отображаем гистограмму
    cv::imwrite("Hist_Abs.png", histogram_abs);
    cv::imwrite("Hist_SIFT.png", histogram_sift);
    cv::waitKey(0);

    return 0;
}
