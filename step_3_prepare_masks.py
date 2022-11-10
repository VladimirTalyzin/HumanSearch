from json import load
from os import path

# noinspection PyProtectedMember
from cv2 import imwrite, IMWRITE_PNG_COMPRESSION
from numpy import zeros, uint8
from skimage import draw
from tqdm import tqdm

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

jsonMasksFile = "region_data.json"
resultPath = "true_masks"

# Размеры масок
HEIGHT = 256
WIDTH  = 256

# Считываем файл с разметкой данных
with open(path.join(root, jsonMasksFile)) as jsonFile:
    masksData = load(jsonFile)


# Создать маску по данным из файла разметок для указанного кусочка фотографии
def generateMask(regions):
    if len(regions) == 0:
        return None
    mask = zeros((HEIGHT, WIDTH), dtype = uint8)

    # Для всех данных из кусочка фотографии, заполнить указанные полигоны белым цветом (255)
    for index, regionData in regions.items():
        rowsPixels, columnsPixels = draw.polygon([min(y, 255) for y in regionData["shape_attributes"]["all_points_y"]],
                                                 [min(x, 255) for x in regionData["shape_attributes"]["all_points_x"]])

        mask[rowsPixels, columnsPixels] = 255

    return mask

# Для всех масок из файла с разметкой
for index, maskData in tqdm(masksData.items()):
    mask = generateMask(maskData["regions"])
    # Сохранить полученную маску
    if mask is not None:
        imwrite(path.join(root, resultPath, maskData["filename"].replace(".jpg", ".png")), mask, [IMWRITE_PNG_COMPRESSION, 9])