from PIL import Image, ImageDraw
import numpy as np


class imageSeparator:
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.non_black_white_masks = []
        self.corners_list = []
        self.vertical_segments = []
        self.load_image_rgb()

    def load_image_rgb(self):
        img = Image.open(self.imagePath).convert("RGB")
        self.rgb_image = img
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

    def isWhitePixel(self, pixel):
        return np.mean(pixel) > 220

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
        self.white_pixels = np.apply_along_axis(
            self.isWhitePixel, axis=2, arr=self.img_array
        )  # shape: (H, W)

        text_rows = np.any(self.white_pixels != True, axis=1)  # shape: (H,)

        text_rows_indices = np.where(text_rows)[0]

        return text_rows_indices

    def find_text_cols(self, segment):
        # Добавить проверку на наличие self.white_pixels

        top = segment[0]
        bottom = segment[-1]

        white_pixels = self.white_pixels[top:bottom, :]

        text_cols = np.any(white_pixels != True, axis=0)  # shape: (W,)

        text_cols_indices = np.where(text_cols)[0]

        print("text_cols_indices", text_cols_indices)

        return text_cols_indices

    def separate(self, separateHeight, separateWidth=10):
        text_rows_indices = self.find_text_rows()

        diffs = np.diff(text_rows_indices)
        split_indices = (
            np.where(diffs > separateHeight)[0] + 1
        )  # +1 чтобы делить *после* разрыва
        segments = np.split(text_rows_indices, split_indices)
        self.filtered_segments = [seg for seg in segments if len(seg) >= 4]

        for segment in self.filtered_segments:
            text_cols_indices = self.find_text_cols(segment)

            diffs = np.diff(text_cols_indices)

            split_indices = (
                np.where(diffs > separateWidth)[0] + 1
            )  # +1 чтобы делить *после* разрыва

            print("split_indices", split_indices)

            segments = np.split(text_cols_indices, split_indices)

            print("segments", segments)

            self.vertical_segments.append([seg for seg in segments if len(seg) >= 4])

            # print("text_cols_indices", text_cols_indices)
            # coords = [
            #     (
            #         text_cols_indices[0],
            #         segment[0],
            #     ),
            #     (
            #         text_cols_indices[-1],
            #         segment[-1],
            #     ),
            # ]
            # self.draw_rectangle_by_coordinates(coords)

        # print("filtered_segments", self.filtered_segments)
        print("self.vertical_segments", self.vertical_segments)

    def horizontal_find_text(self, color_dif):
        for segment in self.filtered_segments:

            mask = np.zeros((len(segment), self.width), dtype=bool)

            for i, row_idx in enumerate(segment):
                row = self.img_array[row_idx]  # строка RGB, shape: (width, 3)

                # print("row", row)

                # Находим максимальную и минимальную компоненту RGB для каждого пикселя
                max_rgb = np.max(row, axis=1)
                min_rgb = np.min(row, axis=1)

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

            # for row in block:
            #     print("row", row)

            colored_coords = np.argwhere(block)

            if len(colored_coords) == 0:
                continue

            min_col = np.min(colored_coords[:, 1])
            max_col = np.max(colored_coords[:, 1])

            top_left = (segment_rows[start], min_col)
            bottom_left = (segment_rows[end], min_col)
            top_right = (segment_rows[start], max_col)
            bottom_right = (segment_rows[end], max_col)

            corners_list.append((top_left, bottom_left, top_right, bottom_right))

        self.corners_list.append(corners_list)

    def draw_rectangle_by_coordinates(self, coords):
        image = Image.fromarray(self.img_array.astype("uint8"))
        draw = ImageDraw.Draw(image)

        draw.rectangle(coords, outline="blue", width=2)
        image.show()

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

    def get_collored_segments(self):
        for corners in self.corners_list:
            for i, (top_left, bottom_left, top_right, bottom_right) in enumerate(
                corners
            ):
                print("top_left", top_left)
                print("bottom_right", bottom_right)

                box = (top_left[1], top_left[0], bottom_right[1], bottom_right[0])

                print("box", box)

                cropped = self.rgb_image.crop(box)

                save_path = f"crop_{i}.jpg"
                cropped.save(save_path)

                print(f"Сохранено: {save_path}")

    def vertical_segment_separate(self):
        for i, non_black_white_segment in enumerate(self.non_black_white_masks, 0):
            self.find_color_regions(non_black_white_segment, self.filtered_segments[i])

    def crop_vertical_segments(self, segment, index):
        for i, verticatl_segment in enumerate(self.vertical_segments[index]):
            box = (
                verticatl_segment[0] - 5,
                segment[0],
                verticatl_segment[-1] + 5,
                segment[-1],
            )

            print("horizontla segment box", box)

            cropped = self.rgb_image.crop(box)

            save_path = f"segments/h_segment_{index}_v_segment_{i}.jpg"
            cropped.save(save_path)

            print(f"Сохранено: {save_path}")

    def crop_horizontal_segments(self):
        for i, segment in enumerate(self.filtered_segments):
            self.crop_vertical_segments(segment, i)


separator = imageSeparator("zayavlenie.png")


# separator.showRGBTable()

# Разделить изображение на горизональные сегменты по белым пропускам (больше 18 пикселей)
separator.separate(separateHeight=10)

separator.crop_horizontal_segments()

# # Находим маску цветных пикселей
# separator.horizontal_find_text(80)  # Обучаемый параметр

# #
# separator.vertical_segment_separate()

# # Рисуем обводку для наглядности
# highlighted_img = separator.draw_color_regions_on_image()
# highlighted_img.show()


# for corners in separator.corners_list:
#     print("corners", corners)

# separator.get_collored_segments()
