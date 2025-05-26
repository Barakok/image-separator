from PIL import Image
import numpy as np


def first_text_line_height(image_path, threshold=200):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    # Логическая маска: True — это "текст" (тёмные пиксели)
    text_mask = arr < threshold

    # Суммируем по строкам (оси 1) — сколько текстовых пикселей в каждой строке
    row_sums = np.sum(text_mask, axis=1)

    # Находим первую и последнюю строку, где есть текст
    text_rows = np.where(row_sums > 0)[0]

    if len(text_rows) == 0:
        return 0, None, None  # Нет текста

    # Смотрим первую строку текста
    start = text_rows[0]

    # Ищем, где эта первая строка заканчивается — пока идут подряд строки с текстом
    for i in range(start + 1, len(row_sums)):
        if row_sums[i] == 0:
            end = i - 1
            break
    else:
        end = text_rows[-1]

    height = end - start + 1
    return height, start, end


# Пример вызова:
height, top, bottom = first_text_line_height("separatorTest.png")
print(f"Высота первой строки текста: {height} пикселей (с {top} по {bottom})")
