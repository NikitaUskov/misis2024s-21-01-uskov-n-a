**Лабораторная 07 (сегментация)**
1. реализовать один метод сегментации на данных лаб.4
2. реализовать метод оценки качества сегментации с использованием эталонной разметки
3. сравнить сегментационный метод и бинаризацию из лаб.4

## **Отчет 07**

Уточнение:

	Данные картинки получаются следующим образом: В методе "createGroundTruth" меняются переменные rows, cols - отвечающие за кол-во кругов в каждой строчке и столбце.
	 
Функциональность:

	Mat segmentUsingKMeans(const Mat& src, int k) – Функция для сегментации с использованием k-means
 
	Mat WatershedSeg(Mat img) -Функция для сегментации с использованием watershed
 
	vector<double> eval(Mat mask, Mat ideal, Mat& result, const string& methodName) - Функция для оценки качества сегментации

### Результат работы программы: 

В отчете будут продемонстрированы картинки с разным кол-вом кругов на них (данные из лаб04): 10, 25, 50, 100, 256. Но сначала мы рассмотрим возможные методы сегментации - Watershed, K-mean на примере для 50 кругов. Также на примере 50 кругов рассмотрим алгоритм работы кода.

#### Полный алгоритм
 Сгенерированная картинка с 50 кругами разных размеров и яркостей, расположенных по возрастанию яркости и размера окружностей:

 ![Orig_50](/prj.lab/lab07/Cir_50/Original.png)

Далее накладываем на картинку размытие с помощью функции: cv::GaussianBlur(img, BlurPic, cv::Size(Ks, Ks), blur);
Затем накладываем шум с помощью функции cv::Mat addNoise(const cv::Mat& orig_im, double noise_sigm). Полученная картинка будет подвережена обработке:

![Noize_50](/prj.lab/lab07/Cir_50/Noisy_Image.png)

Было рассмотрено 2 метода сегментации (кроме бинаризации), визуализация сегментирования. Также они были раскрашены в разные цвета соответственно совпадению разметке, Зеленый - TP, Белый - TN, Красный - FP, Синий - FN:

![Watershed_50](/prj.lab/lab07/Cir_50/Watershed_Evaluation.png) - Watershed

![Kmean_50](/prj.lab/lab07/Cir_50/K-Means_Evaluation.png) - k-mean


Далее применяем оценку качества сегментации по следующим метрикам 
accuracy = (TP + TN) / (TP + FP + FN + TN),
precision = TP / (TP + FP),
recall = TP / (TP + FN).


![Watershed_50](/prj.lab/lab07/Cir_50/Watershed_Segmented.png) - Watershed

![Kmean_50](/prj.lab/lab07/Cir_50/K-Means_Segmented.png) - k-mean

![Binary_50](/prj.lab/lab07/Cir_50/Binary.png) - Binary


#### Сравнение
При сравнении трех методов для 50 кругов, включая бинаризацию, видим самые хорошие показатели у метода обычной бинаризации.

### 10 окружностей

Визуализация сегментирования:
Зеленый - TP, Белый - TN, Красный - FP, Синий - FN:

![Watershed_10](/prj.lab/lab07/Cir_10/Watershed_Evaluation.png) - Watershed

![Kmean_10](/prj.lab/lab07/Cir_10/K-Means_Evaluation.png) - k-mean


Далее применяем оценку качества сегментации:

![Watershed_10](/prj.lab/lab07/Cir_10/Watershed_Segmented.png) - Watershed

![Kmean_10](/prj.lab/lab07/Cir_10/K-Means_Segmented.png) - k-mean

### 25 окружностей

Визуализация сегментирования:
Зеленый - TP, Белый - TN, Красный - FP, Синий - FN:

![Watershed_25](/prj.lab/lab07/Cir_25/Watershed_Evaluation.png) - Watershed

![Kmean_25](/prj.lab/lab07/Cir_25/K-Means_Evaluation.png) - k-mean


Далее применяем оценку качества сегментации:

![Watershed_25](/prj.lab/lab07/Cir_25/Watershed_Segmented.png) - Watershed

![Kmean_25](/prj.lab/lab07/Cir_25/K-Means_Segmented.png) - k-mean

### 100 окружностей

Визуализация сегментирования:
Зеленый - TP, Белый - TN, Красный - FP, Синий - FN:

![Watershed_100](/prj.lab/lab07/Cir_100/Watershed_Evaluation.png) - Watershed

![Kmean_100](/prj.lab/lab07/Cir_100/K-Means_Evaluation.png) - k-mean


Далее применяем оценку качества сегментации:

![Watershed_100](/prj.lab/lab07/Cir_100/Watershed_Segmented.png) - Watershed

![Kmean_100](/prj.lab/lab07/Cir_100/K-Means_Segmented.png) - k-mean

### 256 окружностей

Визуализация сегментирования:
Зеленый - TP, Белый - TN, Красный - FP, Синий - FN:

![Watershed_256](/prj.lab/lab07/Cir_256/Watershed_Evaluation.png) - Watershed

![Kmean_256](/prj.lab/lab07/Cir_256/K-Means_Evaluation.png) - k-mean


Далее применяем оценку качества сегментации:

![Watershed_256](/prj.lab/lab07/Cir_256/Watershed_Segmented.png) - Watershed

![Kmean_256](/prj.lab/lab07/Cir_256/K-Means_Segmented.png) - k-mean

