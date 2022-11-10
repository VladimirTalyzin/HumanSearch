from os import path, listdir
from tqdm import tqdm
# noinspection PyProtectedMember,PyUnresolvedReferences
from cv2 import imread, imwrite, IMWRITE_PNG_COMPRESSION, IMREAD_UNCHANGED
from numpy import zeros, uint8

# Папка с вычисленными маленькими маски test. Предварительно должен быть выполнен step_5_search.py c TRAIN_MODE = False.
masksPath  = "result_masks"

# Папка, куда будут помещены результирующие большие маски с найденными на них людьми.
resultPath = "found_big_masks_test"

# Папка с фотографиями train. Нужна для получения размера будущих больших масок.
picturesPath = "test"

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Составляем словарь найденных масок. Ключ - имя файла, значение - полный путь к файлу
# В папке могут быть preview-файлы для анализа. Их исключаем.
masks = { filename: path.join(masksPath, filename) for filename in listdir(path.join(root, masksPath))
          if filename.endswith(".png") and not filename.endswith("_preview.png")}

# Словарь, в котором будут формироваться большие маски. Ключ - номер файла. Значение - большая маска.
result = {}

# Для всех найденных файлов маленьких масок:
for filename, fullName in tqdm(masks.items()):
    # Из названия файла получаем номер большой маски, а также положение по X и Y
    fileNumber, stringX, stringY = (filename.split(".")[0]).split("_")
    x, y = int(stringX), int(stringY)

    # Если такой большой маски ещё не было в словаре,
    if not fileNumber in result:
        # То загружаем фото с таким же номером из test,
        sourceImage = imread(path.join(root, picturesPath, fileNumber + ".jpg"))
        # чтобы узнать её размер.
        height, width, _ = sourceImage.shape
        # И создаём такую же по размеру маску, заполненную черным (нулями)
        result[fileNumber] = zeros((height, width), dtype = uint8)

    # Считываем маленькую маску
    maskImage = imread(fullName, IMREAD_UNCHANGED)
    # Узнаём её размер
    maskHeight, maskWidth = maskImage.shape

    # Узнаём размер большой маски
    height, width = result[fileNumber].shape

    # Если маленькая маска выходит за пределы большой маски по высоте, то происходит уменьшение размера маленькой маски
    if y + maskHeight > height:
        maskHeight = height - y
        maskImage = maskImage[0:maskHeight,:]

    # Если маленькая маска выходит за пределы большой маски по ширине, то происходит уменьшение размера маленькой маски
    if x + maskWidth > width:
        maskWidth = width - x
        maskImage = maskImage[:,0:maskWidth]

    # Добавляем маленькую маску на большую
    result[fileNumber][y:y + maskHeight, x:x + maskWidth] = maskImage


# Сохраняем все полученные большие маски в целевую папку
for fileNumber, bigImage in tqdm(result.items()):
    imwrite(path.join(root, resultPath, fileNumber + ".png"), bigImage, [IMWRITE_PNG_COMPRESSION, 9])