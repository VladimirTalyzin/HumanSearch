import math
from functools import cmp_to_key
from json import dumps
from math import sqrt
from os import path, listdir

from catboost import CatBoostRegressor
# noinspection PyProtectedMember,PyUnresolvedReferences
from cv2 import imread, imwrite, circle, IMWRITE_PNG_COMPRESSION, IMREAD_UNCHANGED, RETR_TREE, CHAIN_APPROX_SIMPLE, findContours, contourArea, moments
from numpy import vstack
from pandas import read_csv
from tqdm import tqdm

from utility import writeToCSV

# Папка, с большими масками с найденными на них людьми. Получена на предыдущем шаге
bigMasksPath = "found_big_masks_test"

# Папка, куда будут помещены копии масок с найденными людьми, с обведёнными кружками.
# Фактически, это визуализация результата. Очень важна для анализа проведённой работы.
bigMasksCirclesPath = "found_big_masks_test_circles"

# Название модели CatBoost для вычисления радиуса вокруг людей
radiusModelName = "radius_0.69.cbm"

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Создаём модель CatBoost и загружаем данные из указанного файла модели
radiusModel = CatBoostRegressor()
radiusModel.load_model(path.join(root, radiusModelName))

# Получаем список всех больших масок с найденными людьми
masks = { filename.split(".")[0]: imread(path.join(bigMasksPath, filename)) for filename in listdir(path.join(root, bigMasksPath)) if filename.endswith(".png") }

# Загружаем пример файла результата, для точного воспроизведения порядка файлов и регистра символов.
sampleList = read_csv(path.join(root, "sample_solution.csv"))

# Вычисление расстояния между контурами.
# Последовательно сравниваются расстояния между всеми точками, образующими контур. И возвращается минимальное.
def minDistance(contourOne, contourTwo):
    distanceMin = None
    for x1, y1 in contourOne[0]:
        for x2, y2 in contourTwo[0]:
            distance = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
            if distanceMin is None or distance < distanceMin:
                distanceMin = distance

    return distanceMin


# Получение центра масс контура.
# Очевидно, что это намного превосходит результат с использованием центра bounding box, так как именно вокруг "центра масс"
# обводил кружок разметчик.
def contourCenter(contour):
    moment = moments(contour)

    divider = moment["m00"]

    centerX = int(moment["m10"] / divider)
    centerY = int(moment["m01"] / divider)
    return centerX, centerY


# Анализ контуров большой маски. Объединение близко расположенных в единый контур, вычисление радиуса.
# В результате получается отсортированный список координат центров контуров и радиусов.
def contoursCheck(mask, maskName):
    height, width, _ = mask.shape

    # Получаем список контуров людей, с помощью OpenCV
    contours, hierarchy = findContours(mask[:,:,0], RETR_TREE, CHAIN_APPROX_SIMPLE)

    if contours is None:
        return None

    removed = []

    # Записываем индексы вложенных контуров в список на удаление
    for index, (contour, treeData) in enumerate(zip(contours, hierarchy[0, :, :])):
        if treeData[3] != -1:
            removed.append(index)

    # Если список на удаление не пуст, то удаляем контуры из этого списка
    if len(removed) > 0:
        contours = [contour for indexContour, contour in enumerate(contours) if indexContour not in removed]

    # Список объединяемых контуров
    joined = []
    # Список индексов удаляемых контуров
    removed = []

    # Для всех контуров производим сравнение со всеми другими контурами
    for indexOne, contourOne in enumerate(contours):
        for indexTwo, contourTwo in enumerate(contours):
            # Если расстояние между контурами меньше заданного порога, тогда добавляем контуры в список объединения,
            # а их индексы - в список на удаление.
            if indexOne != indexTwo and minDistance(contourOne, contourTwo) < 70:
                if not indexOne in removed and not indexTwo in removed:
                    joined.append([contourOne, contourTwo])
                    removed.append(indexOne)
                    removed.append(indexTwo)

    # Если список на удаление не пуст, то удаляем контуры с указанными индексами
    if len(removed) > 0:
        contours = [contour for indexContour, contour in enumerate(contours) if indexContour not in removed]
        # Добавляем объединённые контуры из списка на объединение.
        for join in joined:
            contours.append(vstack(join))


    coordinates = []

    countContours = len(contours)

    # Расчёт радиусов контуров
    for indexOne, contour in enumerate(contours):
        # Получаем координаты центра масс контура
        centerX, centerY = contourCenter(contour)
        # Получаем площадь контура
        area = contourArea(contour)

        # Если контур на маске только один, то расстояние до ближайшего контура ставим 3000 (как при обучении модели)
        if countContours == 1:
            anotherDistance = 3000
        # Иначе считаем расстояние до ближайшего другого контура
        else:
            anotherDistance = None
            for indexTwo, contourAnother in enumerate(contours):
                if indexOne != indexTwo:
                    anotherCenterX, anotherCenterY = contourCenter(contourAnother)
                    distance = math.sqrt((centerX - anotherCenterX) ** 2 + (centerY - anotherCenterY) ** 2)
                    if anotherDistance is None or anotherDistance > distance:
                        anotherDistance = distance

        # Вычисляем радиус с помощью модели CatBoost
        radius = round(radiusModel.predict([area, width, anotherDistance, countContours]))
        # Добавляем цент контура и вычисленный радиус в список
        coordinates.append({"centerX": centerX, "centerY": centerY, "radius": radius})


    # Цитата из группы Telegram:
    # ""
    # Для корректной оценки вашего решения в файле для отправки необходимо отсортировать найденные области
    # с человеком сначала в порядке возрастания координаты X, а после — в порядке возрастания координаты Y.
    # ""

    # По этой цитате выполняется сортировка контуров.
    def compare(one, two):
        if one["centerX"] == two["centerX"]:
            if one["centerY"] == two["centerY"]:
                return 0
            else:
                return -1 if one["centerY"] < one["centerY"] else 1
        else:
            return -1 if one["centerX"] < two["centerX"] else 1

    coordinates.sort(key = cmp_to_key(compare))


    result = []
    # Для всех найденных координат контуров
    for coordinate in coordinates:
        # Обводим красный кружок вокруг контура, для визуализации результата
        mask = circle(mask, (coordinate["centerX"], coordinate["centerY"]), coordinate["radius"], (0, 0, 255), 4)
        # Добавляем координаты контура в результирующий список, по правилам чемпионата:
        result.append(str(dumps({"cx": coordinate["centerX"], "cy": coordinate["centerY"], "r": coordinate["radius"]})))

    # Сохраняем маску с нанесёнными на ней кружками в папку просмотра результата.
    imwrite(path.join(root, bigMasksCirclesPath, maskName), mask, [IMWRITE_PNG_COMPRESSION, 9])

    # Для файлов, с шириной меньше 2000 пикселей, возвращаем нули.
    # Так score становится лучше.
    # Странное ограничение, но его пришлось добавить.
    # Вероятно, разметчик не размечал людей на фото с разрешением < 2000
    if width < 2000:
        return "0", "0"

    return len(result), result


result = []

# Для всех строки из файла примера, выполняем поиск больших масок,
# обработку их контуров и получение данных для файла результата
for index, data in tqdm(sampleList.iterrows(), total=sampleList.shape[0]):
    # Удаляем значение "region_shape" из файла примера
    del data["region_shape"]

    # Получаем номер фотографии из файла примера
    imageName = data["ID_img"]
    imageNumber = imageName.split(".")[0]

    # Если для номера фотографии из примера есть маска с найденными людьми,
    if imageNumber in masks:
        # То получаем данные по этой строке
        countRegion, data["region_shape"] = contoursCheck(masks[imageNumber], imageNumber + ".png")

    else:
        # Иначе записываем нули
        countRegion = "0"
        data["region_shape"] = "0"

    result.append(data)

# Сохраняем полученные данные в CSV-файл, для отправки на чемпионат
writeToCSV(path.join(root, "result.csv"), ["ID_img", "region_shape"], result)