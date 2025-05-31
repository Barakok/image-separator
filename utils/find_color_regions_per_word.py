import numpy as np


def find_color_regions_per_word(self, mask, segment_rows):
    regions = []
    current_start = None

    for i in range(mask.shape[0]):
        has_color = mask[i].any()

        if has_color and current_start is None:
            current_start = i
        elif not has_color and current_start is not None:
            current_end = i - 1
            regions.append((current_start, current_end))
            current_start = None

    if current_start is not None:
        regions.append((current_start, mask.shape[0] - 1))

    corners_list = []

    for start, end in regions:
        block = mask[start : end + 1, :]  # цветной блок
        row_bounds = []

        # Найдём границы цветных фрагментов в каждой строке
        for i, row in enumerate(block):
            cols = np.where(row)[0]
            if len(cols) == 0:
                row_bounds.append([])
                continue

            # Группировка смежных колонок (одного "слова")
            segments = []
            seg_start = cols[0]
            for j in range(1, len(cols)):
                if cols[j] != cols[j - 1] + 1:
                    segments.append((seg_start, cols[j - 1]))
                    seg_start = cols[j]
            segments.append((seg_start, cols[-1]))  # последний сегмент

            row_bounds.append(segments)

        # Теперь сгруппируем вертикально совпадающие сегменты (как "слова")
        # Храним активные "столбцы" и обновляем
        active_blocks = {}

        for row_idx, segments in enumerate(row_bounds):
            used_keys = set()
            for seg in segments:
                found_key = None
                # Ищем совпадающий активный блок
                for key in active_blocks:
                    if (
                        active_blocks[key][-1][1] == row_idx - 1
                        and active_blocks[key][-1][0] == seg
                    ):
                        found_key = key
                        break
                if found_key is not None:
                    active_blocks[found_key].append((seg, row_idx))
                    used_keys.add(found_key)
                else:
                    new_key = len(active_blocks)
                    active_blocks[new_key] = [(seg, row_idx)]
                    used_keys.add(new_key)

        # Преобразуем в список координат
        for block in active_blocks.values():
            top_row = block[0][1]
            bottom_row = block[-1][1]
            left = block[0][0][0]
            right = block[0][0][1]

            top_left = (segment_rows[start + top_row], left)
            bottom_left = (segment_rows[start + bottom_row], left)
            top_right = (segment_rows[start + top_row], right)
            bottom_right = (segment_rows[start + bottom_row], right)

            corners_list.append((top_left, bottom_left, top_right, bottom_right))

    return corners_list
