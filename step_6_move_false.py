from shutil import move
from json import loads
from os import path, listdir

from pandas import read_csv
from tqdm import tqdm

# Режим работы для первой итерации обучения - True. И для последующих итераций - False.
FIRST_STEP_MODE = True

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Размер изображений для анализа
HEIGHT = 256
WIDTH = 256

# Папка для анализа на наличие ложнопозитивных срабатываний
sourcePath = path.join(root, "result_masks")

if FIRST_STEP_MODE:
    # Папка для переноса ложнопозитивных изображений на первой итерации
    destinationPath = path.join(root, "false_masks")
else:
    # Папка для переноса ложнопозитивных изображений на второй и последующих итерациях
    destinationPath = path.join(root, "false_masks_necessarily")

# Путь к подготовленному файлу данных о позициях людей в train
dataFile = "train_positions.csv"

# Получить словарь всех подходящих файлов в исходной папке
# Такие файлы должны оканчиваться на _preview.png
sourceFiles = { filename.split(".")[0]: path.join(root, sourcePath, filename) for filename in listdir(sourcePath) if filename.endswith("_preview.png") }


# Считываем файл с правильными позициями людей в train
trueData = read_csv(path.join(root, dataFile))

# Словарь bounding box, сгруппированный по названию тренировочного файла
trueBoundingBox = {}

# Добавить для указанного названия фотографии все bounding box указанных в разметке людей
def addBoundingBox(bigImageName, circles):
    for circle in circles:
        if not bigImageName in trueBoundingBox:
            trueBoundingBox[bigImageName] = []

        # делаем запас по радиусу в 10 пикселей
        radius = circle["radius"] + 10
        x = circle["x"]
        y = circle["y"]
        trueBoundingBox[bigImageName].append({"left": x - radius, "top": y - radius, "right": x + radius, "bottom": y + radius})

# Из данных всех людей, указанных в train, формируем словарь допустимых bounding box:
[addBoundingBox(bigImageName.replace(".JPG", ".jpg").replace(".jpg", ""), loads(circles.replace("'", "\"")))
                for bigImageName, circles in zip(trueData["bigImageName"], trueData["circles"])]

# Здесь будет сформирован словарь ложнопозитивных изображений
falseMasksList = {}

# Для всех файлов из исходной папки выполняем проверку на попадание в список bounding box
for filename, fullName in tqdm(sourceFiles.items()):
    fileNumber, x, y, _ = filename.split("_")

    left   = int(x)
    top    = int(y)
    right  = left + WIDTH
    bottom = top + HEIGHT

    # Если для фотографии в разметке указаны люди
    if fileNumber in trueBoundingBox:
        foundIntersect = False
        # Проверяем попадание кусочка фотографии в один из bounding box людей с этой фотографии
        for boundingBox in trueBoundingBox[fileNumber]:
            if ((boundingBox["left"] <= left <= boundingBox["right"]  or boundingBox["left"] <= right  <= boundingBox["right"]) and
                (boundingBox["top"]  <= top  <= boundingBox["bottom"] or boundingBox["top"]  <= bottom <= boundingBox["bottom"])):
                foundIntersect = True
                break

        # Если кусочек не попал ни в один из bounding box, то добавляем в словарь ложных результатов
        if not foundIntersect:
            falseMasksList[filename + ".png"] = fullName

    # Если фотографии не было в разметке, то добавляем в словарь ложных результатов все её части
    else:
        falseMasksList[filename + ".png"] = fullName


# Для всех найденных ложных результатов, переносим их в одну из папок назначений
# false_masks или false_masks_necessarily в зависимости от итерации.
for filename, fullName in tqdm(falseMasksList.items()):
    move(fullName, path.join(root, destinationPath, filename))