## **Лабораторная 08 (визуализация цветового распреления)**

написать консольное приложение для визуализации плотности цветового распределения в linRGB

## **Отчет 08**
	 
Функциональность:

	cv::Mat sRGBtoLinRGB(const cv::Mat& img) – Convert sRGB to Linear RGB
 
	cv::Vec2f change_plane(const cv::Vec3f& vec) – Change the plane of a 3-channel vector

 	cv::Mat prepare_proj_plane() - Prepare the projection plane
 
### Результат работы программы: 

Цвета всех пикселей входного изображения проецируются на плоскость, перпендикулярную главной диагонали цветового куба, за счет этого получается треугольник на итоговых координатах. Далее будут приведены примеры работы программы для некоторых картинок:

1) Входное изображение:

![pic1](/prj.lab/lab08/img/Полоски.png)

Результат:

![rez1](/prj.lab/lab08/img/1.jpg)

2) Входное изображение:

![pic2](/prj.lab/lab08/img/Радуга.jpg)

Результат:

![rez2](/prj.lab/lab08/img/2.jpg)

3) Входное изображение:

![pic3](/prj.lab/lab08/img/Собака.jpg)

Результат:

![rez3](/prj.lab/lab08/img/3.jpg)

4) Входное изображение:

![pic4](/prj.lab/lab08/img/Дедушка.jpg)

Результат:

![rez4](/prj.lab/lab08/img/4.jpg)
