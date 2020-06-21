import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from imutils import paths
from os import makedirs

# расставляет слэши
MODEL = os.path.join('output', 'model.h5')
RESULT = os.path.join('data', 'result')
TEST_IMAGES = os.path.join('data', 'test')
NOISE_IMAGES = os.path.join('data', 'noise_test')

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

if not os.path.exists(NOISE_IMAGES):
    makedirs(NOISE_IMAGES)
model = load_model(MODEL)
test_images = list(paths.list_images(TEST_IMAGES))

for idx, test_image in enumerate(test_images):
    image = cv2.imread(test_image)
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    #image = np.expand_dims(image, axis=-1)
    image = image.astype("float") / 255.0
    image = np.array([image])

    testNoise = np.random.normal(loc=0.1, scale=0.5, size=image.shape)
    image = np.clip(image + testNoise, 0, 1)

    preds = model.predict(image)
    original = (image * 255).astype("uint8")
    recon = (preds * 255).astype("uint8")

    #сохраняем в папку изображения с шумом
    noise_file_path = os.path.join(NOISE_IMAGES, f'{idx}.png')
    cv2.imwrite(noise_file_path, original[0])

    # объединить тестовую картинку с восстановленной
    output = np.hstack([original[0], recon[0]])# итоговое изображение
    file_path = os.path.join(RESULT, f'{idx}.png') # формируем путь для сохранения файла
    cv2.imwrite(file_path, output) # по пути сохраняем файл

