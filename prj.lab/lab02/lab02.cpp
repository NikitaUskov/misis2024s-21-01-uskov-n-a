#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

//создание тестового изображения
cv::Mat generateTestImage(int sq1, int sq2, int cir) {

    int size = 256;
    int squareSize = 209;
    int circleRadius = 83;

    // Создаем изображение
    cv::Mat testImage(size, size, CV_8UC1, cv::Scalar(sq1));

    // Создаем внутренний квадрат
    int squareStart = (size - squareSize) / 2;
    int squareEnd = squareStart + squareSize;
    cv::rectangle(testImage, cv::Point(squareStart, squareStart), cv::Point(squareEnd, squareEnd), cv::Scalar(sq2), -1);

    // Создаем круг
    cv::Point circleCenter(size / 2, size / 2);
    cv::circle(testImage, circleCenter, circleRadius, cv::Scalar(cir), -1);

    return testImage;
}

//Подсчет для гистограммы
std::vector<int> HistCalc(cv::Mat& img) {
    std::vector<int> arrhist(256);
    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            arrhist[static_cast<int>(img.at<uchar>(i, j))]++;
        }
    }
    return arrhist;
}

//создание гистограммы
cv::Mat Histogramma(cv::Mat& img) {

    cv::Mat Hist(img.size(), CV_8UC1, cv::Scalar(230));
    std::vector<int> final;

    cv::normalize(HistCalc(img), final, 0, 230, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < 256; i++) {
        cv::rectangle(Hist, cv::Point(i, 256 - final[i]), cv::Point((i + 1), 256), cv::Scalar(0), -1);
    }

    return Hist;
}

//Горизонтальное соединение
cv::Mat Hconcat(cv::Mat& InImage1, cv::Mat& InImage2, cv::Mat& InImage3, cv::Mat& InImage4) {
    cv::Mat OutImage(InImage1.size().width * 4, InImage1.size().height, CV_8UC1, cv::Scalar(0));
    cv::Mat Temp1(InImage1.size().width * 2, InImage1.size().height, CV_8UC1, cv::Scalar(0));
    cv::Mat Temp2(InImage1.size().width * 2, InImage1.size().height, CV_8UC1, cv::Scalar(0));
    cv::hconcat(InImage1, InImage2, Temp1);
    cv::hconcat(InImage3, InImage4, Temp2);
    cv::hconcat(Temp1, Temp2, OutImage);
    return OutImage;
}

//Вертикальное соединение
cv::Mat Vconcat(cv::Mat& InImage1, cv::Mat& InImage2, cv::Mat& InImage3, cv::Mat& InImage4) {

    cv::Mat OutImage(InImage1.size().width, InImage1.size().height * 4, CV_8UC1, cv::Scalar(0));
    cv::Mat Temp1(InImage1.size().width, InImage1.size().height * 2, CV_8UC1, cv::Scalar(0));
    cv::Mat Temp2(InImage1.size().width, InImage1.size().height * 2, CV_8UC1, cv::Scalar(0));
    cv::vconcat(InImage1, InImage2, Temp1);
    cv::vconcat(InImage3, InImage4, Temp2);
    cv::vconcat(Temp1, Temp2, OutImage);

    return OutImage;
}

//функция зашумления
cv::Mat addNoise(const cv::Mat& orig_im, double noise_sigm) {
    cv::Mat noise(orig_im.size(), CV_8SC1);
    cv::randn(noise, 0, noise_sigm);

    cv::Mat noisy_image;
    orig_im.copyTo(noisy_image);
    noisy_image += noise;

    return noisy_image;
}

//Соединение гистограмм
cv::Mat HistNoise(cv::Mat& InImage1, cv::Mat& InImage2, cv::Mat& InImage3, cv::Mat& InImage4, double sigm) {

    cv::Mat HistN1 = Histogramma(addNoise(InImage1, sigm));
    cv::Mat HistN2 = Histogramma(addNoise(InImage2, sigm));
    cv::Mat HistN3 = Histogramma(addNoise(InImage3, sigm));
    cv::Mat HistN4 = Histogramma(addNoise(InImage4, sigm));

    cv::Mat OutImage = Hconcat(HistN1, HistN2, HistN3, HistN4);

    return OutImage;
}


int main(int argc, char* argv[]) {
    
    //задание начальных параметров
    double sigm1, sigm2, sigm3;
    sigm1 = 3.0; sigm2 = 7.0; sigm3 = 15.0;

    //Создание тестовых картинок
    cv::Mat ImRect1 = generateTestImage(0, 127, 255);
    cv::Mat ImRect2 = generateTestImage(20, 127, 235);
    cv::Mat ImRect3 = generateTestImage(55, 127, 200);
    cv::Mat ImRect4 = generateTestImage(90, 127, 165);
    
    //Получение гистограмм
    cv::Mat TestImage = Hconcat(ImRect1, ImRect2, ImRect3, ImRect4);
    cv::Mat HistTest = Hconcat(Histogramma(ImRect1), Histogramma(ImRect2), Histogramma(ImRect3), Histogramma(ImRect4));
    cv::Mat HistNoise1 = HistNoise(ImRect1, ImRect2, ImRect3, ImRect4, sigm1);
    cv::Mat HistNoise2 = HistNoise(ImRect1, ImRect2, ImRect3, ImRect4, sigm2);
    cv::Mat HistNoise3 = HistNoise(ImRect1, ImRect2, ImRect3, ImRect4, sigm3);

    //Создание основ для гистограмм
    cv::Mat Hist0(256, 256 * 2, CV_8UC1, cv::Scalar(0));
    cv::Mat Hist1(256, 256 * 2, CV_8UC1, cv::Scalar(0));
    cv::Mat Hist2(256, 256 * 2, CV_8UC1, cv::Scalar(0));
    cv::Mat Hist3(256, 256 * 2, CV_8UC1, cv::Scalar(0));

    //Создание пар Картинка-Гистограмма
    cv::vconcat(TestImage, HistTest, Hist0); cv::vconcat(addNoise(TestImage, sigm1), HistNoise1, Hist1);
    cv::vconcat(addNoise(TestImage, sigm2), HistNoise2, Hist2); cv::vconcat(addNoise(TestImage, sigm3), HistNoise3, Hist3);

    cv::Mat Final = Vconcat(Hist0, Hist1, Hist2, Hist3);

    //вывод
    cv::imshow("final", Final);
    cv::imwrite("outputFileName.png", Final);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
