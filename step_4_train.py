import gc
import os
from random import seed, shuffle

import keras.backend as K
import segmentation_models as sm
import tensorflow as tf
from albumentations import Compose, ShiftScaleRotate, HueSaturationValue, RandomGamma
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
from numpy import zeros, uint8
from tqdm.keras import TqdmCallback

from utility import getFiles, createDataset, createSimpleDataset

# У меня возникли проблемы с видеокартой. В процессе обучения через некоторое время
# появлялась ошибка. Поэтому обучение пришлось проводить на процессоре, а видеокарту - отключить
# Если у вас работает видеокарта, то эту строку можно убрать.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Размер картинок для обучения и их масок
IMG_HEIGHT = 256
IMG_WIDTH  = 256

SEED = 515

# Устанавливаем заданный SEED для всех процессов
sm.set_framework('tf.keras')
tf.keras.utils.set_random_seed(SEED)
seed(SEED)

# Указываем корневой каталог запуска. Для localhost - пустой,
# для colab и аналогов - путь к подключённому Google.Drive
root = ""

# Папки с данными для обучения

# Маски сегментации
masksPath = os.path.join(root, "true_masks")
# Изображения для обучения
trainPath = os.path.join(root, "prepared")
# Папка с изображениями, на которых нет людей, изначально заполненная большим количеством изображений.
# Не все они идут на обучение, а только некоторая заданная часть
falseTrainPath = os.path.join(root, "false_masks")
# Папка с изображениями, на которых нет людей. Но обязательная для обучения.
falseTrainPathNecessarily = os.path.join(root, "false_masks_necessarily")


# Создать набор изображений, на которых нет людей. Маски указывать не нужно, все они будут заполнены нулями.
# Можно указать пропорцию useProportion, для того, чтобы добавить только случайную часть от набора масок
# useProportion должно быть <= 1
def createFalseDataset(imagesList, useProportion = 1.0):
    dataLength = len(imagesList)

    if useProportion < 1.0:
        dataLength = int(dataLength * useProportion)
        imagesNames = list(imagesList.keys())
        shuffle(imagesNames)
        imagesNames = imagesNames[:dataLength]
        imagesList = {filename: fullName for filename, fullName in imagesList.items() if filename in imagesNames }

    images = zeros((dataLength, IMG_HEIGHT, IMG_WIDTH, 3), dtype = uint8)

    # Создаём набор данных без масок
    createSimpleDataset(startIndex = 0, imagesList = imagesList, masksList = None, images = images, masks = None, transforms = None,
                        imageWidth = IMG_WIDTH, imageHeight = IMG_HEIGHT)

    # Маски у таких изображений заполнены нулями.
    masks = zeros((dataLength, IMG_HEIGHT, IMG_WIDTH, 1), dtype=uint8)

    return images, masks

# Набор трансформаций обучающей выборки
transformsBasic = Compose(
[
    ShiftScaleRotate(scale_limit = 0.1, rotate_limit = 30, p = 0.3),
])

# Расширенный набор трансформаций дополнительно добавленной обучающей выборки
# Обязателен поворот и небольшое масштабирование
# Также возможно небольшое отклонение по шкале HUE и коэффициенту гаммы

# Здесь можно было бы добавить ещё много чего, например, размытость или нечёткость фотографий,
# Но, как показал анализ train и изменения score на leaderboard, размытые и нечёткие фотографии
# разметчиком не размечались. Так что для исключений таких фотографий было применено
# самое очевидное решение: НЕ добавлять трансформации, связанные с такими искажениями
transformsMore = Compose(
[
    ShiftScaleRotate(scale_limit = 0.1, rotate_limit = 180, p = 1),
    HueSaturationValue(hue_shift_limit = 15, sat_shift_limit = 15, val_shift_limit = 15, p = 0.1),
    RandomGamma(p = 0.1),
])

# Получаем список изображений для обучения
validFilenames = [ filename for filename in os.listdir(masksPath) if filename.endswith("png") and not filename.startswith(".") ]
imagesFilesTrain, maskFilesTrain = getFiles(trainPath, masksPath, validFilenames)

# Получаем список ложнопозитивных изображений после первого обучения (всякие собаки, автомобили и парейдолия с деревьями)
# На первом этапе итерации - пустой
falseTrainFilenames = [ filename for filename in os.listdir(falseTrainPath) if filename.endswith("_preview.png") ]
falseImagesFiles, _ = getFiles(falseTrainPath, None, falseTrainFilenames)

# Получаем список ложнопозитивных изображений, обязательных для добавления в обучение
# На первой и второй итерации - пустой
falseTrainNecessarilyFilenames = [ filename for filename in os.listdir(falseTrainPathNecessarily) if filename.endswith(".png") ]
falseImagesFilesNecessarily, _ = getFiles(falseTrainPathNecessarily, None, falseTrainNecessarilyFilenames)

# Создаём набор данных для обучения. Один раз - из разметки, с применённой базовой трансформацией и ещё два раза с расширенной трансформацией
X_trainTrue, y_trainTrue = createDataset(imagesList = imagesFilesTrain, masksList = maskFilesTrain, transformsBasic = transformsBasic, transformsMore = transformsMore,
                                         useMoreProportion = 2, imageWidth = IMG_WIDTH, imageHeight = IMG_HEIGHT)
# Создаём набор данных ложнопозитивных изображений. Не всех, а только 2/10 части (иначе их будет слишком много и сети станет выгоднее думать,
# что полное отсутствие объектов позволяет получить меньший loss).
X_trainFalse1, y_trainFalse1 = createFalseDataset(imagesList = falseImagesFiles, useProportion = 0.2)
# Создаём набор обязательных ложнопозитивных изображений. Всех, без исключения.
X_trainFalse2, y_trainFalse2 = createFalseDataset(imagesList = falseImagesFilesNecessarily, useProportion = 1)

# Объединяем тензоры для получения итоговой обучающей выборки
X_train = tf.concat([X_trainTrue, X_trainFalse1, X_trainFalse2], axis = 0)
y_train = tf.concat([y_trainTrue, y_trainFalse1, y_trainFalse2], axis = 0)


# Модифицированная функция потери DiceLoss.
# Прописаны значения потери для краевых ситуаций.
# При большом количестве ложнопозитивных обучающих изображений, сети очень хочется начать считать, что объектов вообще нигде нет.
# За это вводится максимальный штраф, в три раза больший, чем при полностью неправильном определении объекта
# Также сети хочется и найти шляпы вместо пней, лысины вместо шишек и подобное. За ложнопозитивное срабатывание также
# добавляется повышенный штраф.
# Правильное срабатывание при пустой маске не определено для классической формулы DiceLoss
# (но обычно решается костылём в виде прибавки поправочного коэффициента (что некрасиво, на мой взгляд)).
# В данной функции просто прописано конкретное значение для этой ситуации - 0, как минимальное значение потери.
def diceLossBlack(targetsPure, inputsPure):
    targets = tf.cast(K.flatten(targetsPure), tf.float32)
    inputs = tf.cast(K.flatten(inputsPure), tf.float32)

    intersection = K.sum(targets * inputs)

    targetsSum = K.sum(targets)
    inputsSum = K.sum(inputs)

    divider = targetsSum + inputsSum

    # если и маска пустая и предсказание пустое, то это лучшее значение потери
    if divider == 0.0:
        return 0.0
    # если маска не пустая, но предсказание пустое, то это худшее значение потери
    elif targetsSum > 0 and inputsSum == 0:
        return 3.0
    # если маска пустая, а предсказание непустое, то добавляется штраф
    elif targetsSum > 0 and inputsSum == 0:
        return 1.4
    # если и маска не пустая и предсказание непустое, тогда классический DiceLoss:
    else:
        return 1 - 2 * intersection / divider


# Также к оценке потери добавляем и классическую оценку энтропии между результатами
# Это немного улучшает обучение
def bceDiceLoss(y_true, y_pred):
    return K.mean(binary_crossentropy(y_true, y_pred)) + diceLossBlack(y_true, y_pred)


# Модель для обучения U-net сети.
# Выбрана EfficientNetB5.
# EfficientNet по опыту работы сейчас выглядит лучшей для обучения сегментации.
# Были проверены все варианты этой сети от B0 до B7. B5 показало лучшие результаты.
# B7 выдавала признаки переобучения. Поэтому была выбрана сеть B5.

# Класс один - человек. Поэтому лучший вариант активации для одного класса - sigmoid
model = sm.Unet("efficientnetb5", classes = 1, activation = "sigmoid")

# Настройки обучения. Собраны в словарь для удобства
settings = \
{
    # Тут классика
    "optimizer": tf.keras.optimizers.Adam(learning_rate = 0.001),
    # Функция потерь. Разработана и описана выше
    "loss": bceDiceLoss,
    # Метрики для просмотра процесса обучения. Интересует правильность определения областей.
    # Это значение показывает функция IOU
    "metrics": [sm.metrics.iou_score],
    # Новое обучение или же загрузить предыдущую обученную модель и дообучить
    # На первой итерации - False, далее - True
    "continue-train": False,
    # В какой файл сохранять модель и читать для дообучения
    "saved-model": "models/efficientnetb5.ckpt",
    # По скольки изображений обучаться одновременно, без пересчёта градиентов.
    # Идеальное значение для обучения равно размеру выборки.
    # Поэтому указываем максимальное значение, которое может выдержать оборудование.
    "batch-size": 24,
    # Количество эпох обучения.
    "epochs": 30,
    # Контроль за обучением. Если IOU тестовой выборки не растёт более пяти итераций подряд, то значит сеть переобучилась
    # Необходимо остановить обучение и вернуться к итерации с лучшим значением IOU
    "callbacks": [EarlyStopping(monitor = 'val_iou_score', mode = 'max', patience = 5, verbose = 0, restore_best_weights = True)]
}

# Если флаг продолжения обучения, тогда считать обученную ранее модель.
if settings["continue-train"]:
    model.load_weights(os.path.join(root, settings["saved-model"])).expect_partial()

# Подготавливаем модель по правилам Keras.
model.compile(settings["optimizer"], settings["loss"], settings["metrics"])

# При подготовке датасетов накапливается много занятой памяти без ссылок на неё.
# Сборщик мусора освободит такую память. Это позволит получить больше ресурсов.
gc.collect()

# Обучить сеть. Настройки берём из заранее указанного справочника settings
# В процессе будет выводиться красивый progress bar.
model.fit(x=X_train,y=y_train, batch_size=settings["batch-size"], epochs=settings["epochs"], verbose = 0, shuffle = True,
          validation_data = (X_trainTrue, y_trainTrue), callbacks = settings["callbacks"] + [TqdmCallback(verbose = 2)])

# Удаляем ссылки на обучающие тензоры
del X_train
del y_train

# Очищаем память без указателей. Иначе её может не хватить на процесс сохранения модели
gc.collect()

# Сохраняем обученную модель в файл
model.save_weights(os.path.join(root, settings["saved-model"]))

# Освобождаем память, занятую Keras.
K.clear_session()