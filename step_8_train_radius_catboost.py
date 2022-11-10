import math

from matplotlib.pyplot import figure, barh, yticks, title, savefig
from numpy import argsort, array
from datetime import datetime
from pandas import read_csv
from json import loads
from os import path, listdir
# noinspection PyProtectedMember,PyUnresolvedReferences
from cv2 import imread, imwrite, findContours, moments, contourArea, IMWRITE_PNG_COMPRESSION, IMREAD_UNCHANGED, RETR_TREE, CHAIN_APPROX_SIMPLE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool

# Указываем SEED, для воспроизводства всех случайных процессов
SEED = 500

# Указываем корневой каталог запуска. Для localhost - пустой, 
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Путь к подготовленному файлу данных о позициях людей в train
dataFile = "train_positions.csv"
# Путь к папке с подготовленными большими масками. Они понадобятся для вычисления площади контуров и нахождения соседних контуров.
bigMasksPath = "found_big_masks_train"


# Получить центр контура, вычисляемый как центр масс
def contourCenter(contour):
    moment = moments(contour)

    divider = moment["m00"]

    centerX = int(moment["m10"] / divider)
    centerY = int(moment["m01"] / divider)
    return centerX, centerY


# Получить список с координатами и площадями контуров
def getContours(mask):
    height, width, _ = mask.shape

    # Находим все контуры на большой маске с помощью OpenCV
    contours, hierarchy = findContours(mask[:, :, 0], RETR_TREE, CHAIN_APPROX_SIMPLE)

    if contours is None:
        return []

    result = []

    for contour, treeData in zip(contours, hierarchy[0, :, :]):
        # Проверяем, не вложенный ли контур. Вложенные - пропускаем.
        if treeData[3] == -1:
            # Получаем площадь контура.	
            area = contourArea(contour)

            if area > 0:
                centerX, centerY = contourCenter(contour)
                # записываем данные о контуре в список
                result.append({"centerX": centerX, "centerY": centerY, "area": area})

    return result


# Получаем справочник по всем большим маскам, где ключ - имя файла маски, а данные - список данных контуров
masksData = { filename: getContours(imread(path.join(bigMasksPath, filename))) for filename in listdir(path.join(root, bigMasksPath)) if filename.endswith(".png") }

# Считываем файл с позициями людей в train
contoursData = read_csv(path.join(root, dataFile))

X = []
y = []


# Поиск ближайшего контура к точке с центром, указанным в данных circle (положение человека и радиус)
# Для этого ближайшего контура получаем площадь.
# Затем ищем ближайший контур к найденному (но не тот же самый).
# И получаем расстояние до этого ближайшего контура. Если контур на маске один, то ставим расстояние 3000.
def getNearestContourAndArea(contours, circle):
    nearestArea = None
    minDistance = None
    foundIndex  = None

    for indexContour, contour in enumerate(contours):
        distance = math.sqrt((circle["x"] - contour["centerX"]) ** 2 + (circle["y"] - contour["centerY"]) ** 2)
        if minDistance is None or minDistance > distance:
            minDistance = distance
            nearestArea = contour["area"]
            foundIndex = indexContour

    minDistance = None
    for indexContour, contour in enumerate(contours):
        if indexContour != foundIndex:
            distance = math.sqrt((circle["x"] - contour["centerX"]) ** 2 + (circle["y"] - contour["centerY"]) ** 2)
            if minDistance is None or minDistance > distance:
                minDistance = distance

    return nearestArea, 3000 if minDistance is None else minDistance


# Для указанного человека из train, получаем данные:
# - площадь контура
# - ширина фотографии, где человек был найден
# - расстояние до ближайшего другого человека
# - количество людей на фото
# - заданный разметчиком радиус
def searchContours(bigImageName, bigImageWidth, countRegions, circles):
    for circle in circles:
        nearestArea, minDistance = getNearestContourAndArea(masksData[bigImageName], circle)

        X.append([nearestArea, bigImageWidth, minDistance, countRegions])
        y.append([circle["radius"]])


# Для всех людей, указанных в train, получаем данные для обучения:
[searchContours(bigImageName.replace(".JPG", ".jpg").replace(".jpg", ".png"), bigImageWidth, countRegions, loads(circles.replace("'", "\"")))
                for bigImageName, bigImageWidth, countRegions, circles in
                    zip(contoursData["bigImageName"], contoursData["bigWidth"], contoursData["countRegions"], contoursData["circles"])]


# отобразить на графике значимость параметров обученной модели
# и сохранить в файл, имя которого будет содержать метку времени
def drawFeatures(model, columns):
    feature_importance = model.feature_importances_
    sorted_idx = argsort(feature_importance)
    figure(figsize=(12, 12))
    barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    yticks(range(len(sorted_idx)), array(columns)[sorted_idx])
    title("Feature Importance")
    savefig("radius_features_" + datetime.now().strftime("%Y-%m-%d-%H-%M") + ".png")


# Разбиваем данные обучения на две части в пропорции 1/10.
# На одной части будем тренировать, на другой - контролировать тренировку.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = SEED)

# формируем данные для обучения CatBoost
featureNames = ["ContourArea", "ImageWidth", "MinDistance", "CountRegions"]
pool_train = Pool(X_train, y_train, feature_names = featureNames)

# формируем данные для контроля обучения CatBoost
pool_test = Pool(X_test, feature_names = featureNames)


# создаём модель CatBoost с параметрами по-умолчанию 
# (за исключением SEED, для воспроизводимости результатов и verbose, 
# чтобы CatBoost не заспамил консоль полезной информацией).
model = CatBoostRegressor(task_type="GPU", random_seed = SEED, verbose = 200)
# Обучаем модель CatBoost
model.fit(pool_train, eval_set = (X_test, y_test))

# Получаем значение качества обучения.
y_prediction = model.predict(pool_test)
score = r2_score(y_test, y_prediction)

# R^2 вышло 0,69. Могло быть и больше, но так явно лучше, чем рандом. Или чем один фиксированный радиус для всех.
# Увеличение score на leaderboard это подтвердило.
print("Train score:", score)

# Сохраняем модель. Она понадобится на последнем шаге работы.
model.save_model("radius_" + datetime.now().strftime("%Y-%m-%d-%H-%M") + ".cbm", format="cbm")

# Для анализа выводим график значимости входных параметров при обучении
drawFeatures(model, featureNames)