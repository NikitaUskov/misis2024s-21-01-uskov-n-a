#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>

// Функция Gray World для выравнивания баланса белого
cv::Mat grayWorld(const cv::Mat& src) {
    if (src.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return cv::Mat();
    }

    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    double avgB = cv::mean(channels[0])[0];
    double avgG = cv::mean(channels[1])[0];
    double avgR = cv::mean(channels[2])[0];

    double avgGray = (avgB + avgG + avgR) / 3.0;

    channels[0] = channels[0] * (avgGray / avgB);
    channels[1] = channels[1] * (avgGray / avgG);
    channels[2] = channels[2] * (avgGray / avgR);

    cv::Mat result;
    cv::merge(channels, result);

    return result;
}

// Функция для ручного автоконтрастирования
cv::Mat autoContrast(const cv::Mat& src) {
    if (src.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return cv::Mat();
    }

    cv::Mat result;
    cv::normalize(src, result, 0, 255, cv::NORM_MINMAX);
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

// Функция для создания изображения с результатами SSIM
cv::Mat createResultImage(double ssimGrayWorld, double ssimAutoContrast) {
    cv::Mat resultImage = cv::Mat::zeros(200, 500, CV_8UC3);
    cv::putText(resultImage, "SSIM Results", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "Original vs Gray World: " + std::to_string(ssimGrayWorld), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(resultImage, "Original vs Auto Contrast: " + std::to_string(ssimAutoContrast), cv::Point(20, 130), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    return resultImage;
}

int main() {
    std::string input = "C:/Users/79194/OneDrive - НИТУ МИСиС/Рисунки/Рандом фотки/Собака.jpg";
    std::string outputGrayWorld = "C:/Users/79194/OneDrive - НИТУ МИСиС/Рисунки/Рандом фотки/Собака_grayworld.jpg";
    std::string outputAutoContrast = "C:/Users/79194/OneDrive - НИТУ МИСиС/Рисунки/Рандом фотки/Собака_autocontrast.jpg";

    // Загружаем изображение
    cv::Mat img = cv::imread(input);

    cv::resize(img, img, cv::Size(500, 500));

    // Применяем функцию Gray World
    cv::Mat resultGrayWorld = grayWorld(img);

    // Сохраняем результат Gray World
    cv::imwrite(outputGrayWorld, resultGrayWorld);

    // Применяем функцию автоконтрастирования
    cv::Mat resultAutoContrast = autoContrast(img);
    

    // Сохраняем результат автоконтрастирования
    cv::imwrite(outputAutoContrast, resultAutoContrast);

    // Вычисляем SSIM между оригиналом и изображениями
    double ssimGrayWorld;
    double ssimAutoContrast;

    cv::Mat ssimMapGrayWorld = computeSSIM(img, resultGrayWorld, ssimGrayWorld);
    cv::Mat ssimMapAutoContrast = computeSSIM(img, resultAutoContrast, ssimAutoContrast);


    // Создаем изображение с результатами SSIM
    cv::Mat resultImage = createResultImage(ssimGrayWorld, ssimAutoContrast);

    // Отображаем результат
    cv::imshow("Original", img);
    cv::imshow("Gray World", resultGrayWorld);
    cv::imshow("Auto Contrast", resultAutoContrast);
    cv::imshow("SSIM Results", resultImage);
    cv::imwrite("SSIMdiff.png", resultImage);

    cv::waitKey(0);

    return 0;
}
