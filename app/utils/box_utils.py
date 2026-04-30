"""
设备框处理工具：合并、命名匹配
"""
import math
from app.core.config import Config


def box_center(box):
    """计算矩形框中心"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_area(box):
    """计算矩形框面积"""
    return (box[2] - box[0]) * (box[3] - box[1])


def box_width(box):
    """计算矩形框宽度"""
    return box[2] - box[0]


def boxes_width_similar(box1, box2, width_ratio_threshold=None):
    """判断两个框的宽度是否相近"""
    if width_ratio_threshold is None:
        width_ratio_threshold = Config.WIDTH_RATIO_THRESHOLD

    w1 = box_width(box1)
    w2 = box_width(box2)

    if w1 == 0 or w2 == 0:
        return False

    width_diff_ratio = abs(w1 - w2) / max(w1, w2)
    return width_diff_ratio <= width_ratio_threshold


def _find_mergeable_boxes(device_boxes, ref_idx, ref_box, ref_width, ref_center_x,
                          width_ratio_threshold, vertical_gap_threshold):
    """寻找可与参考框合并的框"""
    ref_x1, ref_y1, ref_x2, ref_y2 = ref_box
    group_indices = [ref_idx]

    for idx, dev in enumerate(device_boxes):
        if idx == ref_idx:
            continue

        cand_box = dev['box']
        cand_x1, cand_y1, cand_x2, cand_y2 = cand_box
        cand_width = box_width(cand_box)
        cand_center_x = (cand_x1 + cand_x2) / 2

        if not boxes_width_similar(ref_box, cand_box, width_ratio_threshold):
            continue

        x_center_diff = abs(cand_center_x - ref_center_x)
        x_alignment_threshold = ref_width * 0.15

        if x_center_diff > x_alignment_threshold:
            continue

        is_above = cand_y2 < ref_y1
        is_below = cand_y1 > ref_y2

        if is_above or is_below:
            if is_above:
                vertical_dist = ref_y1 - cand_y2
                position = "上方"
            else:
                vertical_dist = cand_y1 - ref_y2
                position = "下方"

            if vertical_dist <= vertical_gap_threshold:
                group_indices.append(idx)
                print(f'[Box Merge]   找到{position}可合并框 (idx={idx}): {cand_box}, '
                      f'宽度={cand_width:.0f}, 垂直距离={vertical_dist:.0f}')

    return group_indices


class BoxUtils:
    """设备框处理工具类"""

    @staticmethod
    def merge_split_device_boxes(device_boxes, width_ratio_threshold=None,
                                 vertical_gap_threshold=None, top_extend=None,
                                 bottom_extend=None, max_attempts=None):
        """合并分裂的 device 矩形框"""
        if width_ratio_threshold is None:
            width_ratio_threshold = Config.WIDTH_RATIO_THRESHOLD
        if vertical_gap_threshold is None:
            vertical_gap_threshold = Config.VERTICAL_GAP_THRESHOLD
        if top_extend is None:
            top_extend = Config.TOP_EXTEND
        if bottom_extend is None:
            bottom_extend = Config.BOTTOM_EXTEND
        if max_attempts is None:
            max_attempts = Config.MAX_ATTEMPTS

        if len(device_boxes) <= 1:
            return device_boxes

        print(f'\n[Box Merge] 合并前: {len(device_boxes)} 个 device 框')

        indexed_boxes = [(idx, dev, box_area(dev['box'])) for idx, dev in enumerate(device_boxes)]
        indexed_boxes.sort(key=lambda x: x[2], reverse=True)

        processed_indices = set()
        successful_attempts = 0

        for ref_idx, ref_dev, ref_area in indexed_boxes:
            if successful_attempts >= max_attempts:
                break

            if ref_idx in processed_indices:
                continue

            print(f'\n[Box Merge] === 尝试第 {successful_attempts + 1}/{max_attempts} '
                  f'个参考框 (idx={ref_idx}) ===')

            ref_box = ref_dev['box']
            ref_x1, ref_y1, ref_x2, ref_y2 = ref_box
            ref_width = box_width(ref_box)
            ref_center_x = (ref_x1 + ref_x2) / 2

            print(f'[Box Merge] 参考框: {ref_box}, 面积={ref_area:.0f}, 宽度={ref_width:.0f}')

            group_indices = _find_mergeable_boxes(
                device_boxes, ref_idx, ref_box, ref_width, ref_center_x,
                width_ratio_threshold, vertical_gap_threshold
            )

            expanded_search = False

            if len(group_indices) <= 1:
                print(f'[Box Merge] 未找到可合并框，扩展搜索范围...')
                expanded_search = True

                extended_ref_box = [
                    ref_x1,
                    ref_y1 - vertical_gap_threshold,
                    ref_x2,
                    ref_y2 + vertical_gap_threshold
                ]

                print(f'[Box Merge] 扩展后的参考框: {extended_ref_box}')

                group_indices = _find_mergeable_boxes(
                    device_boxes, ref_idx, extended_ref_box, ref_width, ref_center_x,
                    width_ratio_threshold, vertical_gap_threshold * 2
                )

            processed_indices.add(ref_idx)
            successful_attempts += 1

            if len(group_indices) > 1:
                print(f'[Box Merge] ✓ 找到 {len(group_indices)} 个可合并的框')

                merged_x1 = min(device_boxes[idx]['box'][0] for idx in group_indices)
                merged_y1 = min(device_boxes[idx]['box'][1] for idx in group_indices)
                merged_x2 = max(device_boxes[idx]['box'][2] for idx in group_indices)
                merged_y2 = max(device_boxes[idx]['box'][3] for idx in group_indices)

                merged_y1 = merged_y1 - top_extend
                merged_y2 = merged_y2 + bottom_extend

                merged_box = [merged_x1, merged_y1, merged_x2, merged_y2]

                merged_text = ''
                for idx in group_indices:
                    text = device_boxes[idx].get('raw_text', '')
                    if text and text != 'Unknown':
                        merged_text = text
                        break

                if not merged_text:
                    merged_text = device_boxes[group_indices[0]].get('raw_text', 'Unknown')

                merged_entry = {
                    'raw_text': merged_text,
                    'box': merged_box,
                    'center': box_center(merged_box),
                    'label': 'device',
                    'merged_from': group_indices
                }

                new_device_boxes = [merged_entry]
                for idx, dev in enumerate(device_boxes):
                    if idx not in group_indices:
                        new_device_boxes.append(dev)

                print(f'[Box Merge] 合并结果: {merged_box} '
                      f'(顶部扩展{top_extend}px, 底部扩展{bottom_extend}px)')
                print(f'[Box Merge] 合并后: {len(new_device_boxes)} 个 device 框\n')

                return new_device_boxes
            else:
                print(f'[Box Merge] ✗ 第 {successful_attempts} 个参考框未找到可合并的框')

                if expanded_search:
                    print(f'[Box Merge] 对该框进行边界扩展: 顶部+{top_extend}px, 底部+{bottom_extend}px')

                    device_boxes[ref_idx]['box'][0] = ref_x1
                    device_boxes[ref_idx]['box'][1] = ref_y1 - top_extend
                    device_boxes[ref_idx]['box'][2] = ref_x2
                    device_boxes[ref_idx]['box'][3] = ref_y2 + bottom_extend

                    device_boxes[ref_idx]['center'] = box_center(device_boxes[ref_idx]['box'])

                    print(f'[Box Merge] 扩展后的框: {device_boxes[ref_idx]["box"]}')

        print(f'\n[Box Merge] 已完成 {successful_attempts} 次有效尝试（目标 {max_attempts} 次）')
        print(f'[Box Merge] 最终结果: {len(device_boxes)} 个 device 框（部分可能已扩展）\n')

        return device_boxes

    @staticmethod
    def match_power_ground_names(power_boxes, ground_boxes, signal_boxes, search_range=40):
        """
        为 power 和 ground 匹配 signalName
        规则：只搜索 box 上下两条边上距离最近的 signalName

        Args:
            power_boxes: power 框列表
            ground_boxes: ground 框列表
            signal_boxes: signalName 框列表
            search_range: 搜索范围（像素）

        Returns:
            tuple: (power_boxes, ground_boxes)
        """

        def find_nearest_signal_on_edges(box, label):
            """在 box 上下边附近寻找最近的 signalName"""
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2

            best_signal = None
            min_dist = float('inf')

            for sig in signal_boxes:
                sx1, sy1, sx2, sy2 = sig['box']
                sig_center_x = (sx1 + sx2) / 2
                sig_center_y = (sy1 + sy2) / 2

                # 检查是否在上下边附近
                is_near_top_edge = (abs(sig_center_y - y1) <= search_range and
                                    x1 - 20 <= sig_center_x <= x2 + 20)
                is_near_bottom_edge = (abs(sig_center_y - y2) <= search_range and
                                       x1 - 20 <= sig_center_x <= x2 + 20)

                if is_near_top_edge or is_near_bottom_edge:
                    # 计算到最近边的距离
                    dist_to_top = abs(sig_center_y - y1)
                    dist_to_bottom = abs(sig_center_y - y2)
                    dist = min(dist_to_top, dist_to_bottom)

                    if dist < min_dist:
                        min_dist = dist
                        best_signal = sig

            return best_signal, min_dist

        # 处理 power boxes
        for power in power_boxes:
            box = power['box']
            best_signal, min_dist = find_nearest_signal_on_edges(box, "Power")

            if best_signal:
                power['raw_text'] = best_signal['raw_text']
                edge = "上边" if abs(best_signal['center'][1] - box[1]) < abs(best_signal['center'][1] - box[3]) else "下边"
                print(f'[DEBUG] Power at {box} → "{power["raw_text"]}" (最近{edge}, 距离: {min_dist:.1f})')
            else:
                power['raw_text'] = "Power"
                print(f'[DEBUG] Power at {box} → "Power" (未找到 signalName)')

        # 处理 ground boxes
        for ground in ground_boxes:
            box = ground['box']
            best_signal, min_dist = find_nearest_signal_on_edges(box, "Ground")

            if best_signal:
                ground['raw_text'] = best_signal['raw_text']
                edge = "上边" if abs(best_signal['center'][1] - box[1]) < abs(best_signal['center'][1] - box[3]) else "下边"
                print(f'[DEBUG] Ground at {box} → "{ground["raw_text"]}" (最近{edge}, 距离: {min_dist:.1f})')
            else:
                ground['raw_text'] = "GND"
                print(f'[DEBUG] Ground at {box} → "GND" (未找到 signalName)')

        return power_boxes, ground_boxes

    @staticmethod
    def match_device_names(device_boxes, signal_boxes, corner_margin=30,
                           top_search_range=50, center_align_threshold=20):
        """为每个 device 匹配一个最合适的 signalName"""
        for dev in device_boxes:
            dx1, dy1, dx2, dy2 = dev['box']
            top_center_x = (dx1 + dx2) / 2

            best_signal = None
            best_priority = 999
            min_dist = float('inf')

            has_inside_signal = False

            for sig in signal_boxes:
                sx1, sy1, sx2, sy2 = sig['box']
                sig_center_x = (sx1 + sx2) / 2
                sig_center_y = (sy1 + sy2) / 2

                is_left_corner = (sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2 and
                                  (sx2 - dx1) < corner_margin and abs(sig_center_y - dy1) < corner_margin * 1.5)

                is_right_corner = (sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2 and
                                   (dx2 - sx1) < corner_margin and abs(sig_center_y - dy1) < corner_margin * 1.5)

                is_above_center = (sy2 < dy1) and (dx1 - 20 <= sig_center_x <= dx2 + 20) and \
                                  (dy1 - top_search_range <= sig_center_y <= dy1)

                is_below_center = (sy1 > dy1) and (dx1 - 20 <= sig_center_x <= dx2 + 20) and \
                                  (dy1 <= sig_center_y <= dy1 + top_search_range)

                is_inside_center_aligned = (sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2 and
                                            abs(sig_center_x - top_center_x) <= center_align_threshold)

                # 检查是否有框内的 signal
                if sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2:
                    has_inside_signal = True

                priority = 999
                dist_to_ref = float('inf')

                if is_left_corner or is_right_corner:
                    priority = 1
                    if is_left_corner:
                        dist_to_ref = math.sqrt((sig_center_x - dx1)**2 + (sig_center_y - dy1)**2)
                    else:
                        dist_to_ref = math.sqrt((sig_center_x - dx2)**2 + (sig_center_y - dy1)**2)
                elif is_above_center:
                    priority = 2
                    dist_to_ref = abs(sig_center_y - dy1)
                elif is_below_center:
                    priority = 3
                    dist_to_ref = abs(sig_center_y - dy1)
                elif is_inside_center_aligned:
                    priority = 4
                    dist_to_ref = abs(sig_center_y - dy1)

                if priority == 999:
                    continue

                if priority < best_priority:
                    best_signal = sig
                    best_priority = priority
                    min_dist = dist_to_ref
                elif priority == best_priority:
                    if dist_to_ref < min_dist:
                        best_signal = sig
                        min_dist = dist_to_ref

            # 如果框内没有 signal，则在下方搜索
            if not has_inside_signal and best_signal is None:
                for sig in signal_boxes:
                    sx1, sy1, sx2, sy2 = sig['box']
                    sig_center_x = (sx1 + sx2) / 2
                    sig_center_y = (sy1 + sy2) / 2

                    # 在 box 下方搜索
                    if (sy1 > dy2) and (dx1 - 30 <= sig_center_x <= dx2 + 30) and \
                            (dy2 < sig_center_y <= dy2 + 80):
                        dist = abs(sig_center_y - dy2)
                        if dist < min_dist:
                            best_signal = sig
                            min_dist = dist
                            best_priority = 5

            if best_signal:
                dev['raw_text'] = best_signal['raw_text']

                sig_box = best_signal['box']
                if best_priority == 1:
                    if sig_box[2] - dx1 < corner_margin:
                        position_info = "左上角(框内)"
                    else:
                        position_info = "右上角(框内)"
                elif best_priority == 2:
                    position_info = "上方(框外)"
                elif best_priority == 3:
                    position_info = "下方(框内近顶)"
                elif best_priority == 4:
                    position_info = f"框内居中(偏移{abs(best_signal['center'][0] - top_center_x):.1f}px)"
                elif best_priority == 5:
                    position_info = "框外下方搜索"

                print(f'[DEBUG] Device at {dev["box"]} → "{dev["raw_text"]}" '
                      f'(位置: {position_info}, 优先级: {best_priority}, 距离: {min_dist:.1f})')
            else:
                dev['raw_text'] = "Unknown"
                print(f'[DEBUG] Device at {dev["box"]} → "Unknown" (未找到信号名)')

        return device_boxes