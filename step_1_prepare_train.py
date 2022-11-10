from pandas import read_csv
from json import loads
from os import path
from tqdm import tqdm
# noinspection PyProtectedMember
from cv2 import imread, imwrite, circle, IMWRITE_PNG_COMPRESSION

from utility import writeToCSV

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Считываем данные для обучения
trainData = read_csv(path.join(root, "train.csv"))

# Оставляем только те строки, где есть люди. И выводим их количество.
filledData = trainData[trainData.count_region > 0]
print("Count images with humans: ", len(filledData))

# Задаём ширину и высоту получаемых изображений с людьми.
# Осмотр train показал, что люди всегда умещаются в эти размеры.
# При этом они выбраны минимально возможными для ускорения обработки и обучения
HEIGHT = 256
WIDTH  = 256

collectData = []

def prepareImage(imageName, countRegion, regionData):
    # Собираем данные о людях. Если радиус > 128 (WIDTH / 2), то ставим 128
    regions = [{"x": int(region["cx"]), "y": int(region["cy"]), "radius": min(int(region["r"]), WIDTH // 2)} for region in loads(regionData.replace("'", ""))]

    # На всякий случай проверяем данные по количеству регионов.
    # Проверка не сработала.
    if countRegion != len(regions):
        print("Invalid regions count!")
        return

    # Получаем bounding box, обведённый вокруг всех людей на одной фотографии
    minX, maxX, minY, maxY = getMinMax(regions)

    # Если полученный bounding box > 256, то необходимо разбивать людей на группы
    if (maxX - minX) > WIDTH or (maxY - minY) > HEIGHT:
        parts = []
        currentPart = []

        # Для всех людей пытаемся добавить других людей, проверяя не стал ли bounding box группы > 256
        for region in regions:
            testMinX, testMaxX, testMinY, testMaxY = getMinMax([*currentPart, region])

            # Если при добавлении человека в группу, bounding box группы стал больше 256, то
            # Добавляем текущую группу, как она есть в список групп, а непомещающегося человека
            # размещаем в следующую группу.
            if (testMaxX - testMinX) > WIDTH or (testMaxY - testMinY) > HEIGHT:
                parts.append(currentPart)
                currentPart = [region]
            # Если bounding box группы не стал больше 256, то добавляем человека в группу.
            else:
                currentPart.append(region)

        # Если последняя группа не пустая, то добавляем её в список групп.
        if len(currentPart) > 0:
            parts.append(currentPart)

        # Для всех найденных групп людей
        for indexPart, part in enumerate(parts):
            # Получаем центр группы
            X, Y = getBox(part)
            # Создаём список людей, вокруг которых нужно обвести круг
            # Это как минимум все люди в группе
            circles = part.copy()

            for region in regions:
                # Проверяем для всех людей на фото (не только тех, что в группе), попадает ли их центр в изображение группы
                if X <= region["x"] <= X + WIDTH and Y <= region["y"] <= Y + HEIGHT:
                    # Если да, то вокруг них тоже будем обводить круг
                    circles.append(region)

            # Создаём изображение группы
            createTrainImage(imageName, X, Y, circles, partNumber = indexPart)

    # Если умещаемся в 256, то сохраняем в изображение с людьми в центре
    else:
        X, Y = getBox(regions)
        createTrainImage(imageName, X, Y, regions, partNumber = 0)


# По данным группы людей, сохранить их в виде части 256х256 общей фотографии
# И ещё одну такую же картинку, но с обведёнными кружками. Для анализа того, как мыслил разметчик данных.
def createTrainImage(imageName, X, Y, circles, partNumber):
    image = imread(path.join(root, "train/" + imageName))
    bigHeight, bigWidth, _ = image.shape
    slice256x256 = image[Y:Y + HEIGHT, X:X + WIDTH]
    example256x256 = slice256x256.copy()

    sliceName = prepareName(imageName, partNumber)

    imageData = {"bigImageName": imageName, "bigWidth": bigWidth, "bigHeight": bigHeight, "sliceName": sliceName, "x": X, "y": Y, "countRegions": 0, "circles": []}

    for circleData in circles:
        circleX = circleData["x"] - X
        circleY = circleData["y"] - Y
        radius  = circleData["radius"]

        # Рисуем красный кружок вокруг людей
        example256x256 = circle(example256x256, (circleX, circleY), radius, (0, 0, 255), 2)
        imageData["circles"].append({"x": circleX + X, "y": circleY + Y, "radius": radius})

    imageData["countRegions"] = len(imageData["circles"])
    collectData.append(imageData)

    # Сохраняем группу и группу с кружками в PNG-файлы
    imwrite(path.join(root, "prepared/" + sliceName), slice256x256, [IMWRITE_PNG_COMPRESSION, 9])
    imwrite(path.join(root, "prepared/example_" + prepareName(sliceName, partNumber)), example256x256, [IMWRITE_PNG_COMPRESSION, 9])


# Получить центр группы людей
def getBox(regions):
    minX, maxX, minY, maxY = getMinMax(regions)
    centerX = int(minX) if minX == maxX else (minX + maxX) // 2
    centerY = int(minY) if minY == maxY else (minY + maxY) // 2
    X = max(centerX - WIDTH // 2, 0)
    Y = max(centerY - HEIGHT // 2, 0)
    return X, Y

# Преобразование имён файлов train для получения итогового имени файла PNG, с индексом _0
# Если на фотографии было несколько групп, то им присвоятся индексы _1, _2 и так далее
def prepareName(imageName, partNumber) -> str:
    return imageName.replace(".JPG", ".jpg").replace(".jpg", "_" + str(partNumber) + ".png")


# Получить bounding box группы людей, с учётом обведённого вокруг них круга
def getMinMax(regions):
    minX = min(regions, key=lambda region: region["x"] - region["radius"])
    maxX = max(regions, key=lambda region: region["x"] + region["radius"])
    minY = min(regions, key=lambda region: region["y"] - region["radius"])
    maxY = max(regions, key=lambda region: region["y"] + region["radius"])

    return minX["x"] - minX["radius"], maxX["x"] + maxX["radius"], minY["y"] - minY["radius"], maxY["y"] + maxY["radius"]


# Выполнить поиск и разбиение на группы для всех фотографий с людьми из train.
for index, data in tqdm(filledData.iterrows(), total=filledData.shape[0]):
    prepareImage(data["ID_img"], data["count_region"], data["region_shape"])


# Сохранить данные групп в файл, который потом понадобится для создания модели определения радиуса
writeToCSV(path.join(root, "train_positions.csv"), ["bigImageName", "bigWidth", "bigHeight", "sliceName", "x", "y", "countRegions", "circles"], collectData)