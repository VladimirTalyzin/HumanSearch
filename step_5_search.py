from os import path, environ
from random import seed
from shutil import copyfile
from pathlib import Path

import cv2
import segmentation_models as sm
import tensorflow as tf
from PIL import Image
from numpy import zeros, expand_dims, uint8
from tqdm import tqdm

from utility import getFiles, createDataset

# Отключить видеокарту для работы на процессоре
environ["CUDA_VISIBLE_DEVICES"]="-1"

# Режим обучения модели если True, либо режим использования модели, если False
TRAIN_MODE = True

# Порог яркости определения класса (от 0 до 255)
THRESHOLD = 220
# Минимальный размер sqrt(150 / π) ≈ 7 пикселей в диаметре
# Анализ разметки показал, что объекты меньшего диаметра не размечались
SQUARE_THRESHOLD = 150

# Файл с моделью
MODEL_FILENAME = "models/efficientnetb5.ckpt"

# Размер изображений для анализа
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Устанавливаем SEED для всех процессов, для получения предсказуемого результата
SEED = 500

sm.set_framework('tf.keras')
tf.keras.utils.set_random_seed(SEED)
seed(SEED)

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

if TRAIN_MODE:
    # Вариант для итерации обучения - поиск по train
    splittedPath = path.join(root, "splitted_train")
else:
    # Вариант для получения решения - поиск по test
    splittedPath = path.join(root, "splitted_test")

# Папка, в которую будут добавляться итоговые картинки
resultPath = path.join(root, "result_masks")

# Создаём модель и загружаем подготовленные веса
model = sm.Unet("efficientnetb5", classes = 1, activation = "sigmoid")
model.load_weights(path.join(root, MODEL_FILENAME)).expect_partial()

# Получаем список всех файлов для поиска маски
imagesCheck, _ = getFiles(splittedPath, None, None)

# Отладочный режим - проверяем только первую тысячу файлов
# imagesCheck = {filename: fullPath for index, (filename, fullPath) in enumerate(imagesCheck.items()) if index < 1000}

# Очищаем папку с результатом
[file.unlink() for file in Path(resultPath).glob("*") if file.is_file()]

# Формируем набор данных из списка файлов. Маски и трансформации при проверке не требуются
X_test, _ = createDataset(imagesList = imagesCheck, masksList = None, transformsBasic = None, transformsMore = None,
                          imageWidth = IMG_WIDTH, imageHeight = IMG_HEIGHT)


# Проверяем контуры полученной маски.
# Если площадь контуров больше SQUARE_THRESHOLD, тогда создаём новую маску, на которой эти контуры залиты белым (255)
# Если подходящих по размеру контуров не было, то вернётся None.
# Необходимо, чтобы отсеять шумы, которые иногда создают модели сегментации.
def contoursCheck(mask):
    # Получаем список контуров
    contours, hierarchy = cv2.findContours(mask[:, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or hierarchy is None:
        return None

    result = None
    # Для всех контуров
    for contour, treeData in zip(contours, hierarchy[0,:,:]):
        # Находим площадь контура
        area = cv2.contourArea(contour)

        if area > SQUARE_THRESHOLD:
            # Если это первый найденный контур на маске, то создаём новую маску
            if result is None:
                result = zeros((IMG_HEIGHT, IMG_WIDTH), dtype = uint8)

            # Заполняем контур белым
            cv2.fillPoly(result, pts = [contour], color = 255)

    return result


# Проверить маску на наличие контуров с цветом выше заданного порога
def checkThreshold(predicted, imageName, threshold):
    # Всё, что выше порога закрашиваем белым, что ниже - чёрным
    predicted[predicted >= threshold] = 255
    predicted[predicted < threshold] = 0

    # Проверяем контуры на порог по площади
    predicted = contoursCheck(predicted)

    # Если такие контуры были найдены
    if predicted is not None:
        predictedImage = Image.fromarray(predicted)
        # Сохраняем итоговую маску в файл, под тем же именем, что было изображение в исходной папке
        predictedImage.save(path.join(root, resultPath, imageName))
        # Для визуального контроля результата копируем в папку результата и само изображение.
        # Чтобы сразу видеть и маску и изображение. Изображение помечается суффиксом _preview
        copyfile(path.join(root, splittedPath, imageName), path.join(root, resultPath, imageName.replace(".png", "_preview.png")))


# Для всех изображений из набора
for imageIndex in tqdm(range(len(imagesCheck))):
    # Получить имя файла изображения, для генерации имени файла результата
    imageName = list(imagesCheck.keys())[imageIndex]

    # Получить предсказание модели
    predictedMask = (model.predict(expand_dims(X_test[imageIndex], axis=0))[0].squeeze() * 255).astype(uint8)

    # Проверить предсказанные данные на соответствие критериям отбора.
    # Если соответствуют, то сохранить маску и изображение в папку результата
    checkThreshold(predictedMask, imageName, THRESHOLD)