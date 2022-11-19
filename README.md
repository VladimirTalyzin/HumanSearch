# HumanSearch
Поиск людей по фотографиям с БПЛА / Search for people in photographs taken from UAVs.

![Распознавание](https://0v.ru/humans/detect-1.png)
![Распознавание](https://0v.ru/humans/detect-3.png)

![Победа 1-е место!](https://0v.ru/humans/diplom-pobeda.jpg)

* Шаг 1: **step_1_prepare_train.py** - подготовить фотографии 256х256 из train, где есть группы людей. Для проведения разметки.
* Шаг 2: **step_2_split_train_and_test.py** - разбить все фотографии train и test на кусочки по 256х256. Получается 104 тыс. картинок для train и 72 тыс. картинок для test.
* Шаг 3: **step_3_prepare_masks.py** - создать маски для обучения сегментации из разметки
* Шаг 4: **step_4_train.py** - обучение модели сегментации **Keras** + **EfficientNetB5**
* Шаг 5: **step_5_search.py** - поиск людей на разбитых кусочках фотографий
* Шаг 6: **step_6_move_false.py** - перенос ложнопозитивных решений в папки для обучения
* Шаг 7: **step_7_prepare_big_train.py** - собрать размеченные фотографии train в большие маски для модели предсказания радиуса
* Шаг 8: **step_8_train_radius_catboost.py** - обучение модели CatBoost для предсказания радиуса
![Параметры CatBoost](https://0v.ru/humans/radius.png)
* Шаг 9: **step_9_prepare_big_test.py** - собрать полученные решения в большие маски результата
* Шаг 10: **step_10_get_result.py** - формирование CSV-файла результата

* Не для чемпионата: **web-site/test_file.py** - реальный поиск людей для любого файла, указанного в командной строке.


Готовая обученная модель: https://0v.ru/humans/model_human_search.zip 
Формат модели: cktp "Checkpoint"


Зависимости:

* pip install keras
* pip install tensorflow
* pip install albumentations
* pip install segmentation-models
* pip install numpy
* pip install scikit-learn
* pip install Pillow
* pip install tqdm
