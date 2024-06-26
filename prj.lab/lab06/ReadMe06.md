## **Лабораторная 06 (детекция окружностей)**
1.	реализовать детектор объектов с использованием Преобразование Хафа
2.	реализовать FROC анализ результатов
3.	сравнить методы 4 и 6 лабораторных



## **Отчет 06**
	 
Функциональность:

	void binaryThreshold(const cv::Mat& src, cv::Mat& dst, int threshold) – функция для бинаризации изображения посредством трешхолда (ползунка).
 
	void evaluateDetections(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) – функция для оценки детекции, подсчитывание следующих параметров: TP (True positive), FP (false positive), FN (false negative). На вход функция принимает вектор найденных кругов, вектор размеченных кругов и значение с трешхолда.
 
	void createGroundTruth(cv::Mat& img, std::vector<cv::Vec3f>& groundTruths) - Функция для создания изображения и разметки ground truth

 	void drawFROC(const std::vector<std::tuple<double, double>>& frocPoints, cv::Mat& frocImg) - для отрисовки результатов FROC

  	cv::HoughCircles(binaryImg, detectedCircles, cv::HOUGH_GRADIENT, 1, binaryImg.rows / 16, param1Value, param2Value, minRad, maxRad) - Метод Хаффа для детекции
   
   	void detectAndEvaluateCircles(const cv::Mat& binaryImg, const cv::Mat& src, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) - Функция для детекции окружностей и оценка детекций

### Результат работы программы: 
	В отчете будут продемонстрированы картинки с разным кол-вом кругов на них: 10, 25, 50, 100, 256. Но сравнение с 4 лабораторной будет только для 100 кругов.

#### Полный алгоритм
 Сгенерированная картинка со 100 кругами разных размеров и яркостей, расположенных по возрастанию яркости и размера окружностей:

 ![Orig_100](/prj.lab/lab06/Cir_100/groundTruth.png)

Далее накладываем на картинку размытие с помощью функции: cv::GaussianBlur(img, BlurPic, cv::Size(Ks, Ks), blur);
Затем накладываем шум с помощью функции cv::Mat addNoise(const cv::Mat& orig_im, double noise_sigm). Полученная картинка будет подвережена обработке:

![Noize_100](/prj.lab/lab06/Cir_100/Noisy_Image.jpg)
 
Также для бинаризации и настройки параметров метода Хафф в качестве GUI был реализован трек бар, одинаковый для всех картинок:

![HaffBar](/prj.lab/lab06/HaffBar.png)

Затем после применения и поиска максимального кол-ва найденных кругов получаем бинаризованную картинку:

![Bin_100](/prj.lab/lab06/Cir_100/binaryImage.png)

При такой бинаризованной картинке получаем следующий результат детекции кругов методом Хаффа:

![Haff_100](/prj.lab/lab06/Cir_100/detectedCircles.png)

В рамках настройки параметров и поиска оптимального решения, каждое изменение параметров FP, FN, TP считывался и изменял значения FROC:

График с расположением точек параметров FPPI, Sensivity.
![FROC_100](/prj.lab/lab06/Cir_100/frocCurve.png)

Видно, что лучший результат изображен на рисунке с детектированными кругами и имеет параметры FPPI = 0 , Sensivity = 0.89

### 10 окружностей

Картинка после детекции:

![Detected_10](/prj.lab/lab06/Cir_10/detectedCircles.png)

Диаграмма FROC результатов (FPPI = 0, Sensivity = 1):

![FROC_10](/prj.lab/lab06/Cir_10/frocCurve.png)

### 25 окружностей

Картинка после детекции:

![Detected_25](/prj.lab/lab06/Cir_25/detectedCircles.png)

Диаграмма FROC результатов (FPPI = 0, Sensivity = 1):

![FROC_25](/prj.lab/lab06/Cir_25/frocCurve.png)

## 50 окружностей

Картинка после детекции:

![Detected_50](/prj.lab/lab06/Cir_50/detectedCircles.png)

Диаграмма FROC результатов (FPPI = 0,02, Sensivity = 0,92):

![FROC_50](/prj.lab/lab06/Cir_50/frocCurve.png)

### 256 окружностей

Картинка после детекции:

![Detected_256](/prj.lab/lab06/Cir_256/detectedCircles.png)

Диаграмма FROC результатов (FPPI = -0.1, Sensivity = 0.76):

![FROC_256](/prj.lab/lab06/Cir_256/frocCurve.png)

В данном условии FPPI получился отрицательным из-за того, что окружности, детектирующие круги были больше и могли "захватить" несколько кругов, что приводило к сбиванию параметров FN, FP.

## Сравнение с lab04

Сравнение методов и результатов будет проводиться только на картинке со 100 кругами, т.к. с меньшим количеством Хафф метод показал максимальные результаты, а с большим сбой по некоторым параметрам.

С помощью Хафф метода были получены следующие результаты: FPPI = 0, Sensivity = 0.89.
При помощи методов 4 лаборатной были получены следующие результаты: TP = 74, FP = 0, FN = 26, => FPPI = 0, Sensivity = 0.74 

Можно сделать вывод, что метод Хаффа работает лучше, чем метод реализованный в lab04
