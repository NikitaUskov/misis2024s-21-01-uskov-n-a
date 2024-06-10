**Лабораторная 06 (детекция окружностей)**
1.	реализовать детектор объектов с использованием Преобразование Хафа
2.	реализовать FROC анализ результатов
3.	сравнить методы 4 и 6 лабораторных



**Отчет 06**
	 
Функциональность:

	void binaryThreshold(const cv::Mat& src, cv::Mat& dst, int threshold) – функция для бинаризации изображения посредством трешхолда (ползунка).
 
	void evaluateDetections(const std::vector<cv::Vec3f>& detections, const std::vector<cv::Vec3f>& groundTruths, double iouThreshold) – функция для оценки детекции, подсчитывание следующих параметров: TP (True positive), FP (false positive), FN (false negative). На вход функция принимает вектор найденных кругов, вектор размеченных кругов и значение с трешхолда.
 
	void createGroundTruth(cv::Mat& img, std::vector<cv::Vec3f>& groundTruths) – функция для разметки изображения для будущего оценивания результативности программы.

Результат работы программы: 
	Изначальная картинка для детектирования кругов:
 ![Img](/prj.lab/lab04/img.png)

Далее накладываем на картинку размытие с помощью функции: cv::GaussianBlur(img, BlurPic, cv::Size(Ks, Ks), blur);
Затем накладываем шум с помощью функции cv::Mat addNoise(const cv::Mat& orig_im, double noise_sigm). Полученная картинка будет подвережена обработке:

 ![FinalyPic](/prj.lab/lab04/ReallyPic.png)
 
Далее необходимо разметить картинку, чтобы потом было с чем сравнивать детектирование:



