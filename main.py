from PIL import Image
import numpy as np


def showRGBTable(r, g, b):
    h, w = r.shape  # Получаем размеры изображения

    for y in range(h):
        for x in range(w):
            pixel = (r[y, x], g[y, x], b[y, x])
            print(f"Pixel at ({y},{x}): R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    img_array = np.array(img)  # shape: (H, W, 3)

    # Разделим на три канала
    red_channel = img_array[:, :, 0]  # Красный
    green_channel = img_array[:, :, 1]  # Зелёный
    blue_channel = img_array[:, :, 2]  # Синий

    return red_channel, green_channel, blue_channel, img_array


def find_pure_white_rows(img_array):
    # Логическая маска: True, если пиксель белый (255,255,255)
    white_pixels = np.all(img_array == 255, axis=2)  # shape: (H, W)

    # Проверяем для каждой строки: все ли пиксели белые?
    white_rows = np.all(white_pixels, axis=1)  # shape: (H,)

    # Получаем индексы строк, где все пиксели белые
    white_row_indices = np.where(white_rows)[0]

    return white_row_indices


def find_text_rows(img_array):
    # Логическая маска: True, если пиксель белый (255,255,255)
    white_pixels = np.all(img_array == 255, axis=2)  # shape: (H, W)

    print("white_pixels", white_pixels)

    # Проверяем для каждой строки: все ли пиксели белые?
    white_rows = np.any(white_pixels != True, axis=1)  # shape: (H,)

    # Получаем индексы строк, где все пиксели белые
    white_row_indices = np.where(white_rows)[0]

    return white_row_indices


r, g, b, img_arr = load_image_rgb("separatorTest.png")


white_rows = find_pure_white_rows(img_arr)

print(white_rows)

text_rows = find_text_rows(img_arr)

print(text_rows)
