#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <string>

//Calculate for histogramm
std::vector<int> HistCalc(cv::Mat& img) {
    std::vector<int> arrhist(256);
    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            arrhist[static_cast<int>(img.at<uchar>(j, i))]++;
        }
    }
    return arrhist;
}

//Create histogramm
cv::Mat Histogramma(cv::Mat& img) {

    cv::Mat Hist(img.size(), CV_8UC1, cv::Scalar(230));
    std::vector<int> final;

    cv::normalize(HistCalc(img), final, 0, 230, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < 256; i++) {
        cv::rectangle(Hist, cv::Point(i, 256 - final[i]), cv::Point((i + 1), 256), cv::Scalar(100), -1);
    }

    return Hist;
}

//Calculate color with quantil
double Quant(cv::Mat& img, double part) {

    std::vector<int> vec = HistCalc(img);
    double sum, sumIm; 
    int i = 0;
    sum = 0.0;
    sumIm = std::accumulate(vec.begin(), vec.end(), 0);
    while ((sum / sumIm) < part) {
        sum += vec[i];
        i++;
    }
    i--;

    return i;
}

//Draw accumulation line
void DrawAccum(cv::Mat& img, cv::Mat& Hist) {
    double sum = 0.0;
    std::vector<int> vec = HistCalc(img);
    for (int i = 0; i < HistCalc(img).size(); i++) {
        sum += vec[i];
        cv::line(Hist, cv::Point(i, 256 - sum / 255.0), cv::Point(i+1, 256 - (sum+vec[i]) / 255.0),cv::Scalar(0));
    }
}

//Contrast function
cv::Mat Contrast(cv::Mat& img, double left, double right) {
    
    cv::Mat new_img(img.size(), CV_8UC1, cv::Scalar(230));
    
    double Cmin, Cmax;
    Cmin = 0.0; Cmax = 255.0;

    new_img = Cmin + (img - left) * ((Cmax - Cmin) / (right - left));

    return new_img;
}

//JoinContrast function
cv::Mat JoinContrast(cv::Mat& colorImage, std::vector<cv::Mat>& JoinColors_contrhists, double quants) {
    
    std::vector<cv::Mat> JoinChannels;
    cv::split(colorImage, JoinChannels);

    double min_quant = 255.0, max_quant = 0.0;
    for (int k = 0; k < JoinChannels.size(); k++) {
        if (min_quant > Quant(JoinChannels[k], quants)) {
            min_quant = Quant(JoinChannels[k], quants);
        }
        if (max_quant < Quant(JoinChannels[k], 1 - quants)) {
            max_quant = Quant(JoinChannels[k], 1 - quants);
        }
    }
    for (int i = 0; i < JoinChannels.size(); i++) {
        //Joint channel contrast
        JoinChannels[i] = Contrast(JoinChannels[i], min_quant, max_quant);
        JoinColors_contrhists[i] = Histogramma(JoinChannels[i]);
        DrawAccum(JoinChannels[i], JoinColors_contrhists[i]);
    }

    cv::Mat ContrJoin_colorImage(256, 256, CV_8UC3);
    cv::merge(JoinChannels, ContrJoin_colorImage);
    return ContrJoin_colorImage;
}

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv,
        "{@image |  | }"
    );
    std::string filename = parser.get<std::string>("@image");
    double quant = 0.1;
    cv::Mat image = cv::imread(filename);//Собака.jpg//Яблоко.jpg//Кровля.jpg
    cv::resize(image, image, cv::Size(256, 256));
    
    cv::Mat grayImage(256, 256, CV_8UC1);
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    cv::Mat colorImage(256, 256, CV_8UC3);
    colorImage = image;

    //Split channel
    std::vector<cv::Mat> channels, DiffChannels;
    cv::split(colorImage, channels);
    DiffChannels = channels;


    //Find border of color
    double col_l, col_r;
    col_l = Quant(grayImage, quant);
    col_r = Quant(grayImage, 1 - quant);
    
    std::vector<double> colors_l;
    std::vector<double> colors_r;
    std::vector<cv::Mat> colors_hists(3); 
    std::vector<cv::Mat> DiffColors_contrhists(3), JoinColors_contrhists(3);

    //Different channel contrast
    for (int i = 0; i < DiffChannels.size(); i++) {
        colors_hists[i] = Histogramma(DiffChannels[i]);
        DrawAccum(DiffChannels[i], colors_hists[i]);
        DiffChannels[i] = Contrast(DiffChannels[i], Quant(DiffChannels[i], quant), Quant(DiffChannels[i], 1 - quant));
        DiffColors_contrhists[i] = Histogramma(DiffChannels[i]);
        DrawAccum(DiffChannels[i], DiffColors_contrhists[i]);
    }


    cv::Mat ContrDiff_colorImage(256, 256, CV_8UC3);
    cv::Mat ContrJoin_colorImage(256, 256, CV_8UC3);
    ContrJoin_colorImage = JoinContrast(colorImage, JoinColors_contrhists, quant);
    cv::merge(DiffChannels, ContrDiff_colorImage);

    //Create contrast
    cv::Mat contr_img(grayImage.size(), CV_8UC1);
    contr_img = Contrast(grayImage, col_l, col_r);

    //Create histogramm
    cv::Mat HistGray = Histogramma(grayImage);
    cv::Mat HistContr = Histogramma(contr_img);

    //Draw the accumulate line
    DrawAccum(grayImage, HistGray);
    DrawAccum(contr_img, HistContr);


    //Concat the images
    cv::Mat gray_stack, color_stackDiff, color_stackJoin, color_stack, FinalImage;
    cv::vconcat(grayImage, contr_img, gray_stack);
    cv::vconcat(colorImage, ContrDiff_colorImage, color_stackDiff);
    cv::vconcat(colorImage, ContrJoin_colorImage, color_stackJoin);


    //Concat the histogramms
    cv::Mat histscolorDiff, histscolorJoin,HistsColorDiff_contr, HistsColorJoin_contr, GrayHists;
    cv::hconcat(colors_hists, histscolorDiff);
    cv::hconcat(colors_hists, histscolorJoin);
    cv::hconcat(DiffColors_contrhists, HistsColorDiff_contr);
    cv::hconcat(JoinColors_contrhists, HistsColorJoin_contr);

    //Final concat
    cv::Mat FinalGray;
    cv::Mat HistCol_Diff, FinalColorDiff;
    cv::Mat HistCol_Join, FinalColorJoin;
    cv::vconcat(HistGray, HistContr, GrayHists);
    cv::hconcat(gray_stack,GrayHists, FinalGray);
    cv::vconcat(histscolorDiff, HistsColorDiff_contr, HistCol_Diff);
    cv::vconcat(histscolorJoin, HistsColorJoin_contr, HistCol_Join);
    cv::cvtColor(HistCol_Diff, HistCol_Diff, cv::COLOR_GRAY2BGR);
    cv::cvtColor(HistCol_Join, HistCol_Join, cv::COLOR_GRAY2BGR);
    cv::hconcat(color_stackDiff, HistCol_Diff, FinalColorDiff);
    cv::hconcat(color_stackJoin, HistCol_Join, FinalColorJoin);

    //Create name of image
    std::vector<std::string> NameHists = {"BlueChannel", "GreenChannel", "RedChannel"};
    for (int i = 0; i < NameHists.size(); i++) {
        cv::putText(FinalColorDiff, NameHists[i], cv::Point(256 * (i + 1) + 20, 20), 1, 0.8, cv::Scalar(0));
        cv::putText(FinalColorJoin, NameHists[i], cv::Point(256 * (i + 1) + 20, 20), 1, 0.8, cv::Scalar(0));
    }
    cv::putText(FinalGray, "GrayImage", cv::Point(256 + 20, 20), 1, 0.8, cv::Scalar(0));

    //Output
    cv::imshow("FinalColorDiff", FinalColorDiff);
    cv::imshow("FinalColorJoin", FinalColorJoin);
    cv::imshow("FinalGray", FinalGray);
    cv::waitKey(0);
    

    return 0;
}
