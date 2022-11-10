import json

import segmentation_models as sm
from os import path, environ
from sys import argv
from math import ceil
from numpy import zeros, uint8, expand_dims
# noinspection PyProtectedMember,PyUnresolvedReferences
from cv2 import imread, imwrite, cvtColor, findContours, contourArea, moments, IMWRITE_PNG_COMPRESSION, COLOR_BGR2RGB, RETR_TREE, CHAIN_APPROX_SIMPLE

environ["CUDA_VISIBLE_DEVICES"]="-1"

WIDTH = 256
HEIGHT = 256

# Минимальная площадь контура
SQUARE_THRESHOLD = 100

# Порог яркости определения класса (от 0 до 255)
THRESHOLD = 200

# Файл с моделью
MODEL_FILENAME = "../models/efficientnetb5.ckpt"

# Путь к папке
root = ""

# Разбить изображение на блоки по 256х256.
# При этом, название файла будет содержать имя файла, из которого он был сделан и координаты X и Y,
# откуда он был разбит
def splitImage(filename, fullName, splittedPath):
    image = imread(fullName)
    imageHeight, imageWidth = image.shape[:2]

    # Список полученных файлов-кусочков
    listSplit = {}

    # Идём по высоте и ширине и разбиваем фотографию на части
    # Если высота или ширина не делится на 256, то дополняем размер
    for indexRow in range(int(ceil(imageHeight / HEIGHT))):
        for indexCol in range(int(ceil(imageWidth / WIDTH))):
            startX  = indexCol * WIDTH
            startY  = indexRow * HEIGHT

            # Если разбитая картинка выходит за края фотографии, то заполняем недостающие части черным цветом (нулями)
            if startX + WIDTH > imageWidth or startY + HEIGHT > imageHeight:
                cropped = zeros((HEIGHT, WIDTH, 3), dtype=uint8)

                if startX + WIDTH > imageWidth and startY + HEIGHT > imageHeight:
                    cropped[0:imageHeight - startY, 0:imageWidth - startX, :] = image[startY:imageHeight, startX:imageWidth]
                elif startX + WIDTH > imageWidth:
                    cropped[0:HEIGHT, 0:imageWidth - startX, :] = image[startY:startY + HEIGHT, startX:imageWidth]
                elif startY + HEIGHT > imageHeight:
                    cropped[0:imageHeight - startY, 0:WIDTH, :] = image[startY:imageHeight, startX:startX + WIDTH]

            # Получаем кусочек фотографии
            else:
                cropped = image[startY:startY + HEIGHT, startX:startX + WIDTH]

            # Получаем имя кусочка
            croppedName = (filename.replace("\\", "/").split("/")[-1]).split(".")[0] + "_" + str(startX) + "_" + str(startY) + ".png"
            croppedFullName = path.join(root, splittedPath, croppedName)
            listSplit[croppedName] = croppedFullName

            # Сохраняем
            imwrite(croppedFullName, cropped, [IMWRITE_PNG_COMPRESSION, 9])

    return listSplit


# Создать набор данных для Keras
def createFileDataset(imagesList):
    images = zeros((len(imagesList), HEIGHT, WIDTH, 3), dtype=uint8)

    for idImage, (filename, fullName) in enumerate(imagesList.items()):
        # Подготавливаем картинку для Keras
        image = imread(fullName)
        image = cvtColor(image, COLOR_BGR2RGB)

        height, width, channels = image.shape

        # Если картинка меньше 256х256, то дополняем недостающую часть черным цветом (нулями)
        if height != HEIGHT or width != WIDTH:
            black = zeros((HEIGHT, WIDTH, 3), dtype="uint8")
            black[0:height, 0:width, :] = image
            image = black

        images[idImage] = image

    return images

# название файла для разбора должно быть указано в командной строке, сразу после названия программы .py
if __name__ == '__main__' and len(argv) > 1:
    file = argv[1]

    # проверяем наличие файла
    if not path.isfile(file):
        print("File " + file + " not found!")
        exit(0)

    # Разбиваем изображение на блоки 256х256
    imagesList = splitImage(file, file, "split")

    sm.set_framework('tf.keras')

    # Создаём модель и загружаем подготовленные веса
    model = sm.Unet("efficientnetb5", classes=1, activation="sigmoid")
    model.load_weights(path.join(root, MODEL_FILENAME)).expect_partial()

    # noinspection PyTypeChecker
    X_test = createFileDataset(imagesList)


    # Получение центра масс контура
    def contourCenter(contour):
        moment = moments(contour)

        divider = moment["m00"]

        centerX = int(moment["m10"] / divider)
        centerY = int(moment["m01"] / divider)
        return centerX, centerY

    # Проверяем контуры полученной маски.
    # Если площадь контуров больше SQUARE_THRESHOLD, тогда создаём новую маску, на которой эти контуры залиты белым (255)
    # Если подходящих по размеру контуров не было, то вернётся None.
    # Необходимо, чтобы отсеять шумы, которые иногда создают модели сегментации.
    def contoursCheck(mask, imageName):
        imageName, x, y = (imageName.split(".")[0]).split("_")
        # Всё, что выше порога закрашиваем белым, что ниже - чёрным
        mask[mask >= THRESHOLD] = 255
        mask[mask < THRESHOLD] = 0

        # Получаем список контуров
        contours, hierarchy = findContours(mask[:, :], RETR_TREE, CHAIN_APPROX_SIMPLE)

        if contours is None or hierarchy is None:
            return []

        result = []
        # Для всех контуров
        for contour, treeData in zip(contours, hierarchy[0, :, :]):
            # Если площадь контура больше порогового значения
            if contourArea(contour) > SQUARE_THRESHOLD:
                centerX, centerY = contourCenter(contour)
                result.append({"x": centerX + int(x), "y": centerY + int(y), "radius": 80})

        return result

    result = []
    # Для всех изображений из набора
    for imageIndex in range(len(imagesList)):
        # Получить имя файла изображения, для генерации имени файла результата
        imageName = list(imagesList.keys())[imageIndex]

        # Получить предсказание модели
        predictedMask = (model.predict(expand_dims(X_test[imageIndex], axis=0))[0].squeeze() * 255).astype(uint8)

        # Проверить предсказанные данные на соответствие критериям отбора.
        # Если соответствуют, то сохранить маску и изображение в папку результата
        contourResult = contoursCheck(predictedMask, imageName)

        if len(contourResult) > 0:
            result = [*result, *contourResult]

    print(json.dumps(result))
