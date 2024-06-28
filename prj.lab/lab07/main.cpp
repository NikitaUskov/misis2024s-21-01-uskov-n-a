#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Функция для добавления шума в изображение
void addNoise(const Mat& src, Mat& dst, double stddev) {
    Mat noise = Mat(src.size(), CV_64F);
    randn(noise, 0, stddev);
    src.convertTo(dst, CV_64F);
    dst += noise;
    dst.convertTo(dst, CV_8U);
}

// Функция для создания изображения и разметки ground truth
void createGroundTruth(Mat& img, vector<Vec3f>& groundTruths) {
    int rows = img.rows / 80;
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

            circle(img, Point(x, y), radius, Scalar(colorValue), -1);
            groundTruths.push_back(Vec3f(x, y, radius));
        }
    }
}

// Функция для сегментации с использованием k-means
Mat segmentUsingKMeans(const Mat& src, int k) {
    Mat samples(src.rows * src.cols, 1, CV_32F);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            samples.at<float>(y + x * src.rows, 0) = src.at<uchar>(y, x);

    Mat labels;
    Mat centers;
    kmeans(samples, k, labels, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    Mat segmented(src.size(), src.type());
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * src.rows, 0);
            segmented.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
        }
    }

    return segmented;
}

// Функция для сегментации с использованием watershed
Mat WatershedSeg(Mat img) {
    Mat img_prep, temp, background, foreground, unknown;
    threshold(img, img_prep, 0, 255, THRESH_OTSU);

    Mat kernel = Mat::ones(3, 3, CV_8U);
    erode(img_prep, temp, kernel);
    dilate(temp, temp, kernel);
    dilate(temp, background, kernel);
    erode(temp, foreground, kernel);
    subtract(background, foreground, unknown);

    Mat markers;
    connectedComponents(foreground, markers);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            markers.at<int>(i, j) = markers.at<int>(i, j) + 1;
            if (unknown.at<uchar>(i, j) == 255) {
                markers.at<int>(i, j) = 0;
            }
        }
    }

    cvtColor(img, temp, COLOR_GRAY2BGR);
    watershed(temp, markers);
    return (markers - 1) > 0;
}

// Функция для оценки качества сегментации
vector<double> eval(Mat mask, Mat ideal, Mat& result, const string& methodName) {
    double accuracy = 0.0, precision = 0.0, recall = 0.0;
    int tp = 0, tn = 0, fp = 0, fn = 0;

    Mat intersec;
    bitwise_and(mask, ideal, intersec);
    tp = sum(intersec)[0];

    Mat intersec_inv;
    bitwise_and(255 - mask, 255 - ideal, intersec_inv);
    tn = sum(intersec_inv)[0];

    fp = sum(mask)[0] / 255 - tp;
    fn = sum(ideal)[0] / 255 - tp;

    if (tp != 0) {
        accuracy = (tp + tn) * 1.0 / (255*(tp + fp + fn + tn));
        precision = tp * 1.0 / (255 *(tp + fp));
        recall = tp * 1.0 / (255*(tp + fn));
    }
    else if (tn != 0) {
        accuracy = (tp + tn) * 1.0 / (tp + fp + fn + tn);
    }

    // Отображение промежуточных изображений
    result = Mat::zeros(mask.size(), CV_8UC3);
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) > 0 && ideal.at<uchar>(i, j) > 0) {
                result.at<Vec3b>(i, j) = Vec3b(0, 255, 0); // TP - Зеленый
            }
            else if (mask.at<uchar>(i, j) == 0 && ideal.at<uchar>(i, j) == 0) {
                result.at<Vec3b>(i, j) = Vec3b(255, 255, 255); // TN - Белый
            }
            else if (mask.at<uchar>(i, j) > 0 && ideal.at<uchar>(i, j) == 0) {
                result.at<Vec3b>(i, j) = Vec3b(0, 0, 255); // FP - Красный
            }
            else if (mask.at<uchar>(i, j) == 0 && ideal.at<uchar>(i, j) > 0) {
                result.at<Vec3b>(i, j) = Vec3b(255, 0, 0); // FN - Синий
            }
        }
    }

    string annotation = "Green - TP, White - TN, Red - FP, Blue - FN";
    putText(result, annotation, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(155, 155, 155), 1);

    // Добавление аннотации
    putText(mask, methodName + " Segmentation", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
    putText(mask, "Accuracy: " + to_string(accuracy), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
    putText(mask, "Precision: " + to_string(precision), Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
    putText(mask, "Recall: " + to_string(recall), Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

    return vector<double>{accuracy, precision, recall};
}

int main() {
    // Создание изображения и эталонной разметки
    int width = 800;
    int height = 1600;
    Mat img = Mat::zeros(height, width, CV_8U);
    vector<Vec3f> groundTruths;
    createGroundTruth(img, groundTruths);

    // Добавление шума
    Mat noisyImg;
    addNoise(img, noisyImg, 30.0);

    // Сегментация с использованием k-means
    int k = 3; // Количество кластеров
    Mat segmentedKMeans = segmentUsingKMeans(noisyImg, k);

    // Сегментация с использованием watershed
    Mat segmentedWatershed = WatershedSeg(noisyImg);

    // Сегментация с использованием бинаризации
    Mat binary;
    threshold(noisyImg, binary, 128, 255, THRESH_BINARY);

    // Оценка качества сегментации k-means
    Mat resultKMeans;
    vector<double> evalResultsKMeans = eval(segmentedKMeans, img, resultKMeans, "K-Means");
    cout << "K-Means Segmentation - Accuracy: " << evalResultsKMeans[0]
        << ", Precision: " << evalResultsKMeans[1]
        << ", Recall: " << evalResultsKMeans[2] << endl;

    // Оценка качества сегментации watershed
    Mat resultWatershed;
    vector<double> evalResultsWatershed = eval(segmentedWatershed, img, resultWatershed, "Watershed");
    cout << "Watershed Segmentation - Accuracy: " << evalResultsWatershed[0]
        << ", Precision: " << evalResultsWatershed[1]
        << ", Recall: " << evalResultsWatershed[2] << endl;

    // Оценка качества бинаризации
    Mat resultBinary;
    vector<double> evalResultsBinary = eval(binary, img, resultBinary, "Binary");
    cout << "Binary Segmentation - Accuracy: " << evalResultsBinary[0]
        << ", Precision: " << evalResultsBinary[1]
        << ", Recall: " << evalResultsBinary[2] << endl;


    // Отображение результатов
    imshow("Original", img);
    imshow("Noisy Image", noisyImg);
    imshow("K-Means Segmented", segmentedKMeans);
    imshow("K-Means Evaluation", resultKMeans);
    imshow("Watershed Segmented", segmentedWatershed);
    imshow("Watershed Evaluation", resultWatershed);
    imshow("Binary", binary);
    imshow("Binary Evaluation", resultBinary);

    imwrite("Original.png", img);
    imwrite("Noisy_Image.png", noisyImg);
    imwrite("K-Means_Segmented.png", segmentedKMeans);
    imwrite("K-Means_Evaluation.png", resultKMeans);
    imwrite("Watershed_Segmented.png", segmentedWatershed);
    imwrite("Watershed_Evaluation.png", resultWatershed);
    imwrite("Binary.png", binary);
    imwrite("Binary_Evaluation.png", resultBinary);

    waitKey(0);

    return 0;
}
