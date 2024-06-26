#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>

// Функция для линейного RGB преобразования
cv::Mat linearizeRGB(const cv::Mat& src) {
    cv::Mat linRGB;
    src.convertTo(linRGB, CV_32F, 1.0 / 255.0);

    cv::Mat linRGBChannels[3];
    cv::split(linRGB, linRGBChannels);

    for (int i = 0; i < 3; ++i) {
        cv::pow(linRGBChannels[i], 1.9, linRGBChannels[i]); // Gamma correction with gamma=2.2
    }

    cv::merge(linRGBChannels, 3, linRGB);
    return linRGB;
}

// Функция Gray World для выравнивания баланса белого
cv::Mat grayWorld(const cv::Mat& src) {
    if (src.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return cv::Mat();
    }

    cv::Mat linRGB = linearizeRGB(src);
    cv::imwrite("linRGB_grayWorld.png", linRGB); // Сохраняем промежуточное изображение

    std::vector<cv::Mat> channels;
    cv::split(linRGB, channels);

    double avgB = cv::mean(channels[0])[0];
    double avgG = cv::mean(channels[1])[0];
    double avgR = cv::mean(channels[2])[0];

    double avgGray = (avgB + avgG + avgR) / 3.0;

    channels[0] = channels[0] * (avgGray / avgB);
    channels[1] = channels[1] * (avgGray / avgG);
    channels[2] = channels[2] * (avgGray / avgR);

    cv::Mat result;
    cv::merge(channels, result);

    cv::pow(result, 1.0 / 2.2, result); // Reverse gamma correction

    result.convertTo(result, CV_8U, 255.0);
    return result;
}

// Функция для ручного автоконтрастирования
cv::Mat autoContrast(const cv::Mat& src) {
    if (src.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return cv::Mat();
    }

    cv::Mat linRGB = linearizeRGB(src);
    cv::imwrite("linRGB_autoContrast.png", linRGB); // Сохраняем промежуточное изображение

    cv::Mat result;
    cv::normalize(linRGB, result, 0, 1, cv::NORM_MINMAX);

    cv::pow(result, 1.0 / 2.2, result); // Reverse gamma correction

    result.convertTo(result, CV_8U, 255.0);
    return result;
}

// Функция для вычисления SSIM
cv::Mat computeSSIM(const cv::Mat& img1, const cv::Mat& img2, double& ssim_value) {
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat I1, I2;
    img1_gray.convertTo(I1, CV_32F);
    img2_gray.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);

    sigma1_2 -= mu1_2;
    sigma2_2 -= mu2_2;
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    ssim_value = cv::mean(ssim_map)[0];

    return ssim_map;
}

// Функция для вычисления PSNR
double computePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat s1;
    cv::absdiff(img1, img2, s1); // |img1 - img2|
    s1.convertTo(s1, CV_32F);    // разница должна быть в формате float
    s1 = s1.mul(s1);             // |img1 - img2|^2

    cv::Scalar s = cv::sum(s1);  // сумма всех элементов

    double sse = s.val[0] + s.val[1] + s.val[2]; // сумма квадратов ошибок

    if (sse == 0) {
        return 0; // изображения идентичны
    }
    else {
        double mse = sse / (double)(img1.channels() * img1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

// Функция для создания изображения с результатами SSIM и PSNR
cv::Mat createResultImage(double ssimGrayWorld, double ssimAutoContrast, double psnrGrayWorld, double psnrAutoContrast) {
    cv::Mat resultImage = cv::Mat::zeros(300, 600, CV_8UC3);
    cv::putText(resultImage, "Quality Metrics", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "Original vs Gray World:", cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "SSIM: " + std::to_string(ssimGrayWorld), cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "PSNR: " + std::to_string(psnrGrayWorld), cv::Point(20, 140), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    cv::putText(resultImage, "Original vs Auto Contrast:", cv::Point(20, 190), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "SSIM: " + std::to_string(ssimAutoContrast), cv::Point(20, 220), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "PSNR: " + std::to_string(psnrAutoContrast), cv::Point(20, 250), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    return resultImage;
}

int main() {
    std::string input = "C://opencv/misis2024s-21-01-uskov-n-a/prj.lab/lab09/Leaves/Листья.jpg";

    // Загружаем изображение
    cv::Mat img = cv::imread(input);
    if (img.empty()) {
        std::cerr << "Error: Unable to load input image!" << std::endl;
        return -1;
    }

    cv::resize(img, img, cv::Size(500, 500));

    // Применяем функцию Gray World
    cv::Mat resultGrayWorld = grayWorld(img);

    // Сохраняем результат Gray World
    cv::imwrite("resultGrayWorld.png", resultGrayWorld);

    // Применяем функцию автоконтрастирования
    cv::Mat resultAutoContrast = autoContrast(img);

    // Сохраняем результат автоконтрастирования
    cv::imwrite("outputAutoContrast.png", resultAutoContrast);

    // Вычисляем SSIM между оригиналом и изображениями
    double ssimGrayWorld;
    double ssimAutoContrast;

    cv::Mat ssimMapGrayWorld = computeSSIM(img, resultGrayWorld, ssimGrayWorld);
    cv::Mat ssimMapAutoContrast = computeSSIM(img, resultAutoContrast, ssimAutoContrast);

    // Вычисляем PSNR между оригиналом и изображениями
    double psnrGrayWorld = computePSNR(img, resultGrayWorld);
    double psnrAutoContrast = computePSNR(img, resultAutoContrast);

    // Создаем изображение с результатами SSIM и PSNR
    cv::Mat resultImage = createResultImage(ssimGrayWorld, ssimAutoContrast, psnrGrayWorld, psnrAutoContrast);

    // Отображаем результат
    cv::imshow("Original", img);
    cv::imshow("Gray World", resultGrayWorld);
    cv::imshow("Auto Contrast", resultAutoContrast);
    cv::imshow("Quality Metrics", resultImage);
    cv::imwrite("QualityMetrics.png", resultImage);

    cv::waitKey(0);

    return 0;
}
