import os
import cv2
from numpy import array, zeros, expand_dims, uint8
from tqdm import tqdm
from random import shuffle
from csv import writer

# Получить словарь с файлами из папки. Где ключ - имя файла, а данные - полный путь к файлу
# Если указан список validNames, то отфильтровываются те файлы, которых нет в этом списке
def getFiles(trainData, resultData, validNames):
    if resultData is not None:
        masks = { filename: os.path.join(resultData, filename)
                  for filename in os.listdir(resultData) if filename.endswith(".png") and (validNames is None or filename in validNames) }
    else:
        masks = None

    images = { filename: os.path.join(trainData, filename)
               for filename in os.listdir(trainData) if filename.endswith(".png") and (validNames is None or filename in validNames) }

    return images, masks


# Создать набор данных по указанному словарю файлов изображений и словарю масок.
# К набору данных можно применить набор трансформации Albumentations
def createSimpleDataset(startIndex, imagesList, masksList, images, masks, transforms, imageWidth, imageHeight):
    for idImage, (filename, fullName) in enumerate(tqdm(imagesList.items())):
        # Если список масок указан, то считываем и подготавливаем маску для Keras
        if masksList is not None:
            mask = cv2.imread(masksList[filename])
            mask = mask[:, :, 0]
            mask = expand_dims(mask, axis=-1)
        # Иначе считаем, что маски нет
        else:
            mask = None

        # Подготавливаем картинку для Keras
        image = cv2.imread(fullName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channels = image.shape

        # Если картинка меньше 256х256, то дополняем недостающую часть черным цветом (нулями)
        if height != imageHeight or width != imageWidth:
            black = zeros((imageHeight, imageWidth, 3), dtype="uint8")
            black[0:height, 0:width, :] = image
            image = black

        # Применяем трансформации Albumentations
        if transforms is not None:
            augmented = transforms(image = array(image), mask = mask)
            del image
            image = augmented["image"]
            if mask is not None:
                del mask
                mask = augmented["mask"]

        if mask is not None:
            masks[startIndex + idImage] = mask

        images[startIndex + idImage] = image


# Создать набор данных, из словаря файлов изображений и масок.
# Можно указать трансформацию Albumentations для набора данных в transformsBasic,
# и ещё раз добавить этот же набор данных в количестве частей useMoreProportion,
# указав для него увеличенный набор трансформаций transformsMore
def createDataset(imagesList, masksList, transformsBasic, transformsMore, imageWidth, imageHeight, useMoreProportion = 0.0):
    # Длина базового набора данных
    dataLength = len(imagesList)
    # Длина дополнительного набора данных
    moreSize = int(float(dataLength) * useMoreProportion) if transformsMore is not None and useMoreProportion > 0 else 0
    # Количество изображений для обучения равно сумме длин
    images = zeros((dataLength + moreSize, imageHeight, imageWidth, 3), dtype = uint8)

    # Если словарь файлов масок указан, то создаём соответствующее количеству изображений количество пустых масок
    if masksList is not None:
        masks = zeros((dataLength + moreSize, imageHeight, imageWidth, 1), dtype = bool)
    else:
        masks = None

    startIndex = 0
    # Заполняем изображения и маски файлами из словаря изображений и масок, применив базовую трансформацию Albumentations
    createSimpleDataset(startIndex = startIndex, imagesList = imagesList, masksList = masksList, images = images, masks = masks,
                        transforms = transformsBasic, imageWidth = imageWidth, imageHeight = imageHeight)
    startIndex += dataLength

    # Если useMoreProportion > 1, то необходимо добавить этот же набор данных с расширенной трансформацией transformsMore,
    # столько раз, сколько целых в useMoreProportion
    if moreSize > dataLength:
        for repeatIndex in range(moreSize // dataLength):
            repeatImagesList = list(imagesList.keys())
            moreSize -= dataLength
            shuffle(repeatImagesList)

            allFiles = {filename: fullName for filename, fullName
                        in imagesList.items() if filename in repeatImagesList}

            createSimpleDataset(startIndex = startIndex, imagesList = allFiles, masksList = masksList, images = images, masks = masks,
                                transforms = transformsMore, imageWidth = imageWidth, imageHeight = imageHeight)
            startIndex += dataLength

    # Если оставшаяся часть moreSize > 0, тогда дополнительные данные равны части от основного набора данных
    # Эта часть берётся в отдельный массив, перемешивается, к ней добавляется расширенная трансформация transformsMore,
    # и добавляется в итоговый набор данных
    if 0 < moreSize <= dataLength:
        imagesNames = list(imagesList.keys())
        shuffle(imagesNames)
        moreImagesList = {filename: fullName for filename, fullName in imagesList.items() if filename in imagesNames[:moreSize] }
        createSimpleDataset(startIndex = dataLength, imagesList = moreImagesList, masksList=masksList, images=images, masks=masks,
                            transforms = transformsMore, imageWidth = imageWidth, imageHeight = imageHeight)

    return images, masks


# Сохранение списка словарей в CSV-формат.
def writeToCSV(filename, columns, listOfRows, rawFormat = False):
    with open(filename, "w", encoding = "utf8", newline="") as csvFile:
        csvWriter = writer(csvFile, delimiter = ",")
        csvWriter.writerow(columns)
        for row in listOfRows:
            csvWriter.writerow(row if rawFormat else row.values())