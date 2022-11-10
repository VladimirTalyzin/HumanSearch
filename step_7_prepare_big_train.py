from pandas import read_csv
from numpy import zeros, uint8
from os import path
from tqdm import tqdm
# noinspection PyProtectedMember, PyUnresolvedReferences
from cv2 import imwrite, imread, IMWRITE_PNG_COMPRESSION, IMREAD_UNCHANGED

# Путь к подготовленному файлу данных о позициях людей в train
dataFile = "train_positions.csv"
# Путь к подготовленным маскам train
maskPath = "true_masks"
# Папка, в которой будут сохранены итоговые большие маски train
resultPath = "found_big_masks_train"

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Считываем все данные о позициях
trainData = read_csv(path.join(root, dataFile))

# В этом словаре будут храниться созданные большие маски. Ключ - имя файла, данные - маска
bigMasks = {}

# Размер маленькой маски
MASK_WIDTH  = 256
MASK_HEIGHT = 256


# Добавление маленькой маски на большую маску
def addMaskToBigMask(bigImageName, width, height, sliceName, x, y):
    # Если большой маски для файла с таким именем ещё не существовало, то создаём её
    if not bigImageName in bigMasks:
        bigMasks[bigImageName] = zeros((height, width), dtype = uint8)

    bigMask = bigMasks[bigImageName]

    # Загружаем в память первый канал маленькой маски (все каналы у маски, очевидно, одинаковые).
    maskImage = imread(sliceName)[:,:,0]

    # Если маленькая маска выходит за пределы большой маски по высоте, то происходит уменьшение размера маленькой маски
    if y + MASK_HEIGHT > height:
        maskHeight = height - y
        maskImage = maskImage[0:maskHeight, :]
    else:
        maskHeight = MASK_HEIGHT

    # Если маленькая маска выходит за пределы большой маски по ширине, то происходит уменьшение размера маленькой маски
    if x + MASK_WIDTH > width:
        maskWidth = width - x
        maskImage = maskImage[:, 0:maskWidth]
    else:
        maskWidth = MASK_WIDTH

    # Добавляем маленькую маску на большую
    bigMask[y:y + maskHeight, x:x + maskWidth] = maskImage


# Для всех данных из файла о позициях, выполняем добавление маленьких масок на большую
[addMaskToBigMask(path.join(root, resultPath, bigImageName), bigWidth, bigHeight, path.join(root, maskPath, sliceName), x, y)
                    for bigImageName, bigWidth, bigHeight, sliceName, x, y in zip(trainData["bigImageName"], trainData["bigWidth"], trainData["bigHeight"],
                                                                 trainData["sliceName"], trainData["x"], trainData["y"])]


# Сохраняем все полученные большие маски в целевую папку
for bigImageName, bigMask in tqdm(bigMasks.items()):
    imwrite(bigImageName.replace(".JPG", ".jpg").replace(".jpg", ".png"), bigMask, [IMWRITE_PNG_COMPRESSION, 9])