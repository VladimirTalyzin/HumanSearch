from math import ceil
from os import path, listdir
# noinspection PyProtectedMember
from cv2 import imread, imwrite, IMWRITE_PNG_COMPRESSION
from numpy import zeros, uint8
from tqdm import tqdm

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Ширина и высота разбиения фотографий
WIDTH = 256
HEIGHT = 256

# Папки фотографий и папки результата
trainPath = "train"
splittedTrainPath = "splitted_train"

testPath = "test"
splittedTestPath = "splitted_test"

# Считываем файлы
trainFiles = { filename.split(".")[0]: path.join(root, trainPath, filename) for filename in listdir(trainPath) if filename.endswith(".JPG") or filename.endswith(".jpg") }
testFiles  = { filename.split(".")[0]: path.join(root, testPath,  filename) for filename in listdir(testPath)  if filename.endswith(".JPG") or filename.endswith(".jpg") }

# Разбить изображение на блоки по 256х256.
# При этом, название файла будет содержать имя файла, из которого он был сделан и координаты X и Y,
# откуда он был разбит
def splitImage(filename, fullName, splittedPath):
    image = imread(fullName)
    imageHeight, imageWidth = image.shape[:2]

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
            croppedName = path.join(root, splittedPath, filename + "_" + str(startX) + "_" + str(startY) + ".png")

            # Сохраняем
            imwrite(croppedName, cropped, [IMWRITE_PNG_COMPRESSION, 9])


# Для всех файлов train
for filename, fullName in tqdm(trainFiles.items()):
    splitImage(filename, fullName, splittedTrainPath)

# Для всех файлов test
for filename, fullName in tqdm(testFiles.items()):
    splitImage(filename, fullName, splittedTestPath)