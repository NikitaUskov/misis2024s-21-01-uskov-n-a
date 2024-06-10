**Лабораторная 07 (сегментация)**
1. реализовать один метод сегментации на данных лаб.4
2. реализовать метод оценки качества сегментации с использованием эталонной разметки
3. сравнить сегментационный метод и бинаризацию из лаб.4

**Отчет07**

Функциональность:

	void evaluateDetections(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) – функция для оценки детекции, подсчитывание следующих параметров: TP (True positive), FP (false positive), FN (false negative). На вход функция принимает вектор найденных кругов, вектор размеченных кругов и значение с трешхолда.
 
	void createGroundTruth(cv::Mat& img, std::vector<cv::Vec3f>& groundTruths) – функция для разметки изображения для будущего оценивания результативности программы.

	cv::HoughCircles(binaryImg, detectedCircles, cv::HOUGH_GRADIENT, 1, binaryImg.rows / 16, param1Value, param2Value, 5, 35); - Детектирование кругов методом Хаффа выполняется в этой функции

 	void evaluateFROC(const std::vector<std::vector<cv::Vec3f>>& allDetections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) - функция для оценки результатов детекции по FROC (Free-response Receiver Operating Characteristic)

Результат работы программы: 
	Изначальная картинка для детектирования кругов (из лабораторной 4):

 ![FinalyPic](/prj.lab/lab04/ReallyPic.png)
 
Далее необходимо разметить картинку, чтобы потом было с чем сравнивать детектирование:

![BinTrue](/prj.lab/lab06/BinTrue.png)

Далее применяем метод сегментацию с помощью LoG, функция которого приведена выше:

![BinDet](/prj.lab/lab07/LoG.png)


Также была реализована функция оценки с помощью FROC: И на данных параметрах результат следующий

**FPPI** = 0,2 (FPPI= FN / N, где N - кол-во кругов)

**Sensitivity** = 0,83 ( Sen = TP / (TP + FN) )
