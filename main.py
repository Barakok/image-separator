from PIL import Image, ImageDraw
import numpy as np


class imageSeparator:
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.non_black_white_masks = []
        self.corners_list = []
        self.load_image_rgb()

    def load_image_rgb(self):
        img = Image.open(self.imagePath).convert("RGB")
        self.img_array = np.array(img)  # shape: (H, W, 3)

        # Разделим на три канала
        self.red_channel = self.img_array[:, :, 0]  # Красный
        self.green_channel = self.img_array[:, :, 1]  # Зелёный
        self.blue_channel = self.img_array[:, :, 2]  # Синий

        self.height, self.width = self.red_channel.shape  # Получаем размеры изображения

    def showRGBTable(self):
        for y in range(self.height):
            for x in range(self.width):
                pixel = (
                    self.red_channel[y, x],
                    self.green_channel[y, x],
                    self.blue_channel[y, x],
                )
                print(f"Pixel at ({y},{x}): R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")

    def find_pure_white_rows(self):
        # Логическая маска: True, если пиксель белый (255,255,255)
        white_pixels = np.all(self.img_array == 255, axis=2)  # shape: (H, W)

        # Проверяем для каждой строки: все ли пиксели белые?
        white_rows = np.all(white_pixels, axis=1)  # shape: (H,)

        # Получаем индексы строк, где все пиксели белые
        white_row_indices = np.where(white_rows)[0]

        print("white", white_row_indices)

    def find_text_rows(self):
        # Логическая маска: True, если пиксель белый (255,255,255)
        white_pixels = np.all(self.img_array == 255, axis=2)  # shape: (H, W)

        text_rows = np.any(white_pixels != True, axis=1)  # shape: (H,)

        text_rows_indices = np.where(text_rows)[0]

        return text_rows_indices

    def separate(self, separateHeight):
        text_rows_indices = self.find_text_rows()
        diffs = np.diff(text_rows_indices)
        split_indices = (
            np.where(diffs > separateHeight)[0] + 1
        )  # +1 чтобы делить *после* разрыва
        segments = np.split(text_rows_indices, split_indices)
        self.filtered_segments = [seg for seg in segments if len(seg) >= 4]

        # print("filtered_segments", self.filtered_segments)

    def horizontal_find_text(self, color_dif):
        for segment in self.filtered_segments:

            mask = np.zeros((len(segment), self.width), dtype=bool)

            for i, row_idx in enumerate(segment):
                row = self.img_array[row_idx]  # строка RGB, shape: (width, 3)

                # print("row", row)

                # Находим максимальную и минимальную компоненту RGB для каждого пикселя
                max_rgb = np.max(row, axis=1)
                min_rgb = np.min(row, axis=1)

                print("max_rgb", max_rgb)
                print("min_rgb", min_rgb)

                # Пиксель считается цветным, если разница между RGB-каналами > 10
                is_colored = (
                    max_rgb - min_rgb
                ) > color_dif  # color_dif - обучаемый параметр?

                mask[i] = is_colored

            self.non_black_white_masks.append(mask)

        # print("self.non_black_white_masks", self.non_black_white_masks[-1])

        # for line in self.non_black_white_masks[-1]:
        #     print("line", line)

    def find_corners(self, mask, segment_rows):
        # mask: bool array (rows, width)
        # segment_rows: array с номерами строк в исходном изображении

        rows, width = mask.shape

        # Получаем индексы всех цветных пикселей
        colored_coords = np.argwhere(mask)  # array с координатами (row_idx, col_idx)
        if len(colored_coords) == 0:
            return None  # если цветных пикселей нет

        # Найдем минимальный и максимальный столбец среди цветных пикселей
        min_col = np.min(colored_coords[:, 1])
        max_col = np.max(colored_coords[:, 1])

        # Среди пикселей в min_col найдём верхнюю (минимальный индекс строки) и нижнюю (максимальный) точки
        min_col_pixels = colored_coords[colored_coords[:, 1] == min_col]
        top_left_row_idx = np.min(min_col_pixels[:, 0])
        bottom_left_row_idx = np.max(min_col_pixels[:, 0])

        # Аналогично для max_col
        max_col_pixels = colored_coords[colored_coords[:, 1] == max_col]
        top_right_row_idx = np.min(max_col_pixels[:, 0])
        bottom_right_row_idx = np.max(max_col_pixels[:, 0])

        # Получаем реальные координаты в исходном изображении (номера строк)
        top_left = (segment_rows[top_left_row_idx], min_col)
        bottom_left = (segment_rows[bottom_left_row_idx], min_col)
        top_right = (segment_rows[top_right_row_idx], max_col)
        bottom_right = (segment_rows[bottom_right_row_idx], max_col)

        print("top_left", top_left)
        print("bottom_left", bottom_left)
        print("top_right", top_right)
        print("bottom_right", bottom_right)

        return top_left, bottom_left, top_right, bottom_right

    def find_color_regions(self, mask, segment_rows):
        regions = []
        current_start = None

        print("segment_rows", segment_rows)

        for i in range(mask.shape[0]):
            has_color = mask[i].any()

            if has_color and current_start is None:
                current_start = i  # начало цветного блока

            elif not has_color and current_start is not None:
                current_end = i - 1  # конец цветного блока
                regions.append((current_start, current_end))
                current_start = None

        # Если в конце остался незавершённый блок
        if current_start is not None:
            regions.append((current_start, mask.shape[0] - 1))

        corners_list = []

        print("regions", regions)

        for start, end in regions:
            block = mask[start : end + 1, :]  # цветной блок

            print("start", start)
            print("end", end)
            print("block", block)
            print("block shape", block.shape)

            # for row in block:
            #     print("row", row)

            colored_coords = np.argwhere(block)

            print("colored_coords", colored_coords)

            if len(colored_coords) == 0:
                continue

            min_col = np.min(colored_coords[:, 1])
            max_col = np.max(colored_coords[:, 1])

            print("min_col", min_col)
            print("max_col", max_col)

            min_col_rows = colored_coords[colored_coords[:, 1] == min_col][:, 0]
            max_col_rows = colored_coords[colored_coords[:, 1] == max_col][:, 0]

            print("min_col_rows", min_col_rows)
            print("min_col_rows", max_col_rows)

            top_left = (segment_rows[start], min_col)
            bottom_left = (segment_rows[end], min_col)
            top_right = (segment_rows[start], max_col)
            bottom_right = (segment_rows[end], max_col)

            corners_list.append((top_left, bottom_left, top_right, bottom_right))

        for corners in corners_list:
            print("corners", corners)

        self.corners_list.append(corners_list)

    def draw_color_regions_on_image(self):
        """
        np_img — numpy-массив RGB-изображения (shape: H x W x 3)
        regions — список кортежей: [(top_left, bottom_left, top_right, bottom_right), ...]
        """
        # Конвертируем numpy-массив в изображение
        image = Image.fromarray(self.img_array.astype("uint8"))
        draw = ImageDraw.Draw(image)

        for corners in self.corners_list:
            for i, (top_left, bottom_left, top_right, bottom_right) in enumerate(
                corners
            ):
                # Вычисляем границы прямоугольника
                min_x = min(top_left[1], bottom_left[1])
                max_x = max(top_right[1], bottom_right[1])
                min_y = min(top_left[0], top_right[0])
                max_y = max(bottom_left[0], bottom_right[0])

                # Нарисовать прямоугольник (красным цветом)
                draw.rectangle(
                    [(min_x, min_y), (max_x, max_y)], outline="blue", width=2
                )

        return image

    def vertical_segment_separate(self):
        for i, non_black_white_segment in enumerate(self.non_black_white_masks, 0):
            self.find_color_regions(non_black_white_segment, self.filtered_segments[i])


separator = imageSeparator("zayavlenie.png")


separator.showRGBTable()

# Разделить изображение на горизональные сегменты по белым пропускам (больше 18 пикселей)
separator.separate(separateHeight=19)

# Находим маску цветных пикселей
separator.horizontal_find_text(80)  # Обучаемый параметр

#
separator.vertical_segment_separate()

# Рисуем обводку для наглядности
highlighted_img = separator.draw_color_regions_on_image()
highlighted_img.show()
