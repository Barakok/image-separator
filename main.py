from PIL import Image, ImageDraw
import numpy as np


class imageSeparator:
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.non_black_white_masks = []
        self.corners_list = []
        self.vertical_segments = []
        self.load_image_rgb()
        self.segments = []

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

        return text_cols_indices

    def separate(self, separateHeight, separateWidth=10):
        text_rows_indices = self.find_text_rows()

        diffs = np.diff(text_rows_indices)
        split_indices = (
            np.where(diffs > separateHeight)[0] + 1
        )  # +1 чтобы делить *после* разрыва
        segments = np.split(text_rows_indices, split_indices)
        self.filtered_segments = [seg for seg in segments if len(seg) >= 4]

        for h_segment in self.filtered_segments:

            text_cols_indices = self.find_text_cols(h_segment)

            diffs = np.diff(text_cols_indices)

            split_indices = (
                np.where(diffs > separateWidth)[0] + 1
            )  # +1 чтобы делить *после* разрыва

            segments = np.split(text_cols_indices, split_indices)

            vertical_segments = [seg for seg in segments if len(seg) >= 4]

            self.vertical_segments.append(vertical_segments)

            for v_segment in vertical_segments:

                left = v_segment[0] - 5
                right = v_segment[-1] + 1 + 5
                top = h_segment[0] - 5
                bottom = h_segment[-1] + 1 + 5

                segment_res = self.img_array[top:bottom, left:right, :]

                self.segments.append(segment_res)

        self.save_segments()

    def save_segments(self):
        for i, segment in enumerate(self.segments):
            img = Image.fromarray(segment)
            filename = f"segment_{i}.png"
            img.save(f"segments/{filename}")

    def find_colored_text(self, color_dif):
        for segment in self.segments:

            shape = segment.shape
            mask = np.zeros((shape[0], shape[1]), dtype=bool)

            for i, row in enumerate(segment):
                max_rgb = np.max(row, axis=1)
                min_rgb = np.min(row, axis=1)

                is_colored = (
                    max_rgb - min_rgb
                ) > color_dif  # color_dif - обучаемый параметр?

                mask[i] = is_colored

            self.non_black_white_masks.append(mask)

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

        for start, end in regions:
            block = mask[start : end + 1, :]  # цветной блок

            # for row in block:
            #     print("row", row)

            colored_coords = np.argwhere(block)

            if len(colored_coords) == 0:
                continue

            min_col = np.min(colored_coords[:, 1])
            max_col = np.max(colored_coords[:, 1])

            top_left = (start, min_col)
            bottom_left = (end, min_col)
            top_right = (start, max_col)
            bottom_right = (end, max_col)

            corners_list.append((top_left, bottom_left, top_right, bottom_right))

        self.corners_list.append(corners_list)

    def draw_rectangle_by_coordinates(self, coords):
        image = Image.fromarray(self.img_array.astype("uint8"))
        draw = ImageDraw.Draw(image)

        draw.rectangle(coords, outline="blue", width=2)
        image.show()

    def draw_color_regions_on_image(self):

        for i, corners in enumerate(self.corners_list):
            if len(corners) > 0:
                image = Image.fromarray(self.segments[i])
                draw = ImageDraw.Draw(image)

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

                # image.show()

    def get_colored_segments(self):
        for cor_idx, corners in enumerate(self.corners_list):
            segment_image = Image.fromarray(self.segments[cor_idx])

            for i, (top_left, bottom_left, top_right, bottom_right) in enumerate(
                corners
            ):
                box = (top_left[1], top_left[0], bottom_right[1], bottom_right[0])

                cropped = segment_image.crop(box)

                save_path = f"segments/corners_{cor_idx}_colored_segment_{i}.jpg"
                cropped.save(save_path)

                print(f"Сохранено: {save_path}")

    def vertical_segment_separate(self):
        for i, non_black_white_segment in enumerate(self.non_black_white_masks, 0):
            self.find_color_regions(non_black_white_segment, self.segments[i])

    def crop_vertical_segments(self, segment, index):
        for i, verticatl_segment in enumerate(self.vertical_segments[index]):
            box = (
                verticatl_segment[0] - 5,
                segment[0],
                verticatl_segment[-1] + 5,
                segment[-1],
            )

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

# separator.crop_horizontal_segments()

# # Находим маску цветных пикселей
separator.find_colored_text(80)  # Обучаемый параметр

# #
separator.vertical_segment_separate()

# Рисуем обводку для наглядности
highlighted_img = separator.draw_color_regions_on_image()


for corners in separator.corners_list:
    print("corners", corners)

separator.get_colored_segments()
