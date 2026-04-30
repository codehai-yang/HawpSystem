"""
主程序：HAWP线段检测 + BFS图搜索 + OCR信号名 → 连接关系表
"""
import os
import sys
import json
import csv
from collections import defaultdict
import base64

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import Config
from app.services.hawp_detector import HawpDetector
from app.services.graph_builder import GraphBuilder
from app.services.connection_finder import ConnectionFinder
from app.utils.box_utils import BoxUtils
from app.utils.visualizer import Visualizer
from app.services.pdf_service import process_single_page_image

def box_intersects_device(signal_box_expanded, device_box):
    """判断两个矩形框是否相交"""
    return not (signal_box_expanded[2] < device_box[0] or
                signal_box_expanded[0] > device_box[2] or
                signal_box_expanded[3] < device_box[1] or
                signal_box_expanded[1] > device_box[3])

def point_in_box(point, box):
    """判断点是否在矩形框内"""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

def merge_junctions_in_signal_box(junctions, adj, hawp_lines, signal_boxes, device_boxes,
                                  expand_distance=30, merge_threshold=50.0):
    """
    合并 signalName box 内方向一致的 junction

    Args:
        junctions: 节点列表
        adj: 邻接表
        hawp_lines: 线段列表
        signal_boxes: signalName 框列表
        device_boxes: device 框列表
        expand_distance: signalName box 扩展距离
        merge_threshold: junction 合并阈值（像素）

    Returns:
        tuple: (新的 junctions, 新的 adj)
    """
    jmap = {j['id']: j for j in junctions}

    # 构建线段到端点的映射
    line_endpoints = []
    for idx, line in enumerate(hawp_lines):
        line_endpoints.append((line['p1'], line['p2']))

    # 找到每个 junction 相关的线段索引
    junction_to_lines = defaultdict(list)
    for line_idx, (p1, p2) in enumerate(line_endpoints):
        # 查找 p1 和 p2 对应的 junction
        for jid, j in jmap.items():
            if abs(j['x'] - p1[0]) < merge_threshold and abs(j['y'] - p1[1]) < merge_threshold:
                junction_to_lines[jid].append(line_idx)
            if abs(j['x'] - p2[0]) < merge_threshold and abs(j['y'] - p2[1]) < merge_threshold:
                junction_to_lines[jid].append(line_idx)

    merged_count = 0

    for sig_box in signal_boxes:
        box = sig_box['box']

        # 扩展 box
        expanded_box = [
            box[0] - expand_distance,
            box[1] - expand_distance,
            box[2] + expand_distance,
            box[3] + expand_distance
        ]

        # 检查是否与两个 device 相交
        intersecting_devices = []
        for dev in device_boxes:
            if box_intersects_device(expanded_box, dev['box']):
                intersecting_devices.append(dev)

        # 如果与两个 device 相交，跳过（已经添加了辅助线）
        if len(intersecting_devices) == 2:
            continue

        # 找到 box 内的所有 junction
        junctions_in_box = []
        for jid, j in jmap.items():
            if point_in_box((j['x'], j['y']), box):
                junctions_in_box.append(jid)

        if len(junctions_in_box) < 2:
            continue

        print(f'  [Junction合并] signalName "{sig_box.get("raw_text", "")}" at {box}: '
              f'找到 {len(junctions_in_box)} 个 junction')

        # 按方向分组：水平和垂直
        horizontal_groups = []
        vertical_groups = []

        for jid in junctions_in_box:
            related_lines = junction_to_lines.get(jid, [])

            if not related_lines:
                continue

            # 分析这些线段的方向
            has_horizontal = False
            has_vertical = False

            for line_idx in related_lines:
                p1, p2 = line_endpoints[line_idx]
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])

                if dx > dy * 2:  # 水平线段
                    has_horizontal = True
                elif dy > dx * 2:  # 垂直线段
                    has_vertical = True

            if has_horizontal and not has_vertical:
                horizontal_groups.append(jid)
            elif has_vertical and not has_horizontal:
                vertical_groups.append(jid)
            # 如果既有水平又有垂直，不处理

        # 合并水平方向的 junction（Y 坐标相近的）
        if horizontal_groups:
            horizontal_groups.sort(key=lambda jid: jmap[jid]['y'])

            current_group = [horizontal_groups[0]]
            for i in range(1, len(horizontal_groups)):
                prev_jid = current_group[-1]
                curr_jid = horizontal_groups[i]

                if abs(jmap[curr_jid]['y'] - jmap[prev_jid]['y']) < merge_threshold * 2:
                    current_group.append(curr_jid)
                else:
                    if len(current_group) > 1:
                        _merge_junction_group(junctions, adj, jmap, current_group, "水平")
                        merged_count += 1
                    current_group = [curr_jid]

            if len(current_group) > 1:
                _merge_junction_group(junctions, adj, jmap, current_group, "水平")
                merged_count += 1

        # 合并垂直方向的 junction（X 坐标相近的）
        if vertical_groups:
            vertical_groups.sort(key=lambda jid: jmap[jid]['x'])

            current_group = [vertical_groups[0]]
            for i in range(1, len(vertical_groups)):
                prev_jid = current_group[-1]
                curr_jid = vertical_groups[i]

                if abs(jmap[curr_jid]['x'] - jmap[prev_jid]['x']) < merge_threshold * 2:
                    current_group.append(curr_jid)
                else:
                    if len(current_group) > 1:
                        _merge_junction_group(junctions, adj, jmap, current_group, "垂直")
                        merged_count += 1
                    current_group = [curr_jid]

            if len(current_group) > 1:
                _merge_junction_group(junctions, adj, jmap, current_group, "垂直")
                merged_count += 1

    print(f'\n共合并 {merged_count} 组 junction')
    return junctions, adj

def _merge_junction_group(junctions, adj, jmap, group_jids, direction):
    """合并一组 junction 为一个"""
    if len(group_jids) < 2:
        return

    # 计算平均坐标
    avg_x = sum(jmap[jid]['x'] for jid in group_jids) / len(group_jids)
    avg_y = sum(jmap[jid]['y'] for jid in group_jids) / len(group_jids)

    # 创建新的 junction
    new_jid = max(jmap.keys()) + 1
    new_junction = {'id': new_jid, 'x': avg_x, 'y': avg_y}
    junctions.append(new_junction)
    jmap[new_jid] = new_junction

    # 合并邻接关系
    new_neighbors = set()
    for jid in group_jids:
        neighbors = adj.get(jid, set())
        new_neighbors.update(neighbors)

    # 移除旧的 junction ID
    new_neighbors -= set(group_jids)

    # 更新邻接表
    adj[new_jid] = new_neighbors
    for neighbor in new_neighbors:
        if neighbor in adj:
            for old_jid in group_jids:
                adj[neighbor].discard(old_jid)
            adj[neighbor].add(new_jid)

    # 删除旧的 junction
    for jid in group_jids:
        if jid in adj:
            del adj[jid]
        if jid in jmap:
            del jmap[jid]

    print(f'    合并 {len(group_jids)} 个{direction}junction -> 新 junction {new_jid} at ({avg_x:.1f}, {avg_y:.1f})')


def add_auxiliary_lines_for_signal_boxes(hawp_lines, signal_boxes, device_boxes, expand_distance=50):
    """
    为 signalName box 添加辅助线段

    Args:
        hawp_lines: HAWP 检测到的线段列表
        signal_boxes: signalName 框列表
        device_boxes: device 框列表
        expand_distance: signalName box 扩展距离

    Returns:
        list: 原始线段 + 辅助线段的完整列表
    """
    auxiliary_lines = []
    added_count = 0

    for sig_box in signal_boxes:
        box = sig_box['box']
        center = sig_box['center']

        # 计算 box 的宽度和高度
        width = box[2] - box[0]
        height = box[3] - box[1]

        # 扩展 box
        expanded_box = [
            box[0] - expand_distance,
            box[1] - expand_distance,
            box[2] + expand_distance,
            box[3] + expand_distance
        ]

        # 统计与扩展后的 signal box 相交的 device 数量
        intersecting_devices = []
        for dev in device_boxes:
            if box_intersects_device(expanded_box, dev['box']):
                intersecting_devices.append(dev)

        # 如果恰好与两个 device 相交
        if len(intersecting_devices) == 2:
            # 判断最长边是水平还是垂直
            if width >= height:
                # 水平方向最长，在左右两边的中点连线
                left_mid_x = expanded_box[0]
                left_mid_y = (expanded_box[1] + expanded_box[3]) / 2
                right_mid_x = expanded_box[2]
                right_mid_y = (expanded_box[1] + expanded_box[3]) / 2

                auxiliary_lines.append({
                    'p1': (left_mid_x, left_mid_y),
                    'p2': (right_mid_x, right_mid_y),
                    'score': 1.0,
                    'is_auxiliary': True
                })
                added_count += 1
                print(f'  [辅助线] signalName "{sig_box.get("raw_text", "")}" at {box}: '
                      f'添加水平辅助线 ({left_mid_x:.1f}, {left_mid_y:.1f}) -> '
                      f'({right_mid_x:.1f}, {right_mid_y:.1f})')
            else:
                # 垂直方向最长，在上下两边的中点连线
                top_mid_x = (expanded_box[0] + expanded_box[2]) / 2
                top_mid_y = expanded_box[1]
                bottom_mid_x = (expanded_box[0] + expanded_box[2]) / 2
                bottom_mid_y = expanded_box[3]

                auxiliary_lines.append({
                    'p1': (top_mid_x, top_mid_y),
                    'p2': (bottom_mid_x, bottom_mid_y),
                    'score': 1.0,
                    'is_auxiliary': True
                })
                added_count += 1
                print(f'  [辅助线] signalName "{sig_box.get("raw_text", "")}" at {box}: '
                      f'添加垂直辅助线 ({top_mid_x:.1f}, {top_mid_y:.1f}) -> '
                      f'({bottom_mid_x:.1f}, {bottom_mid_y:.1f})')

    print(f'\n共添加 {added_count} 条辅助线段')

    # 返回原始线段 + 辅助线段
    return hawp_lines + auxiliary_lines
def parse_objects(objects):
    """解析 PDF Service 返回的 objects"""
    device_boxes = []
    ground_boxes = []
    power_boxes = []
    signal_boxes = []

    for obj in objects:
        pts = obj.get('points', [])
        if not pts:
            continue

        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]

        if len(pts) == 2:
            box = [min(xs), min(ys), max(xs), max(ys)]
        elif len(pts) >= 4:
            box = [min(xs), min(ys), max(xs), max(ys)]
        else:
            continue

        label = obj.get('label', '')
        text = obj.get('text', '').strip() if label == 'signalName' else ''

        entry = {
            'raw_text': text,
            'box': box,
            'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            'label': label,
        }

        if label == 'device':
            device_boxes.append(entry)
        elif label == 'ground':
            ground_boxes.append(entry)
        elif label == 'power':
            power_boxes.append(entry)
        elif label == 'signalName':
            signal_boxes.append(entry)

    print(f'\nParsed: {len(device_boxes)} devices, {len(ground_boxes)} grounds, '
          f'{len(power_boxes)} powers, {len(signal_boxes)} signals')

    # 合并分裂的 device 框
    device_boxes = BoxUtils.merge_split_device_boxes(device_boxes)

    # 执行 device 名字匹配
    device_boxes = BoxUtils.match_device_names(device_boxes, signal_boxes)

    # 设置 power 和 ground 的默认名字
    for power in power_boxes:
        power['raw_text'] = "KL30"

    for ground in ground_boxes:
        ground['raw_text'] = "GND"

    for dev in device_boxes:
        print(f'[DEBUG] Device at {dev["box"]} assigned name: "{dev["raw_text"]}"')

    for pwr in power_boxes:
        print(f'[DEBUG] Power at {pwr["box"]} assigned name: "{pwr["raw_text"]}"')

    for gnd in ground_boxes:
        print(f'[DEBUG] Ground at {gnd["box"]} assigned name: "{gnd["raw_text"]}"')

    return device_boxes, ground_boxes, power_boxes, signal_boxes



def save_results(connections, out_dir, base):
    """保存结果为 CSV 和 JSON"""
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, base + '_connections.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['信号名', '设备A', '设备B'])
        for c in connections:
            w.writerow([c['signal'], c['from_device'], c['to_device']])
    print(f'CSV: {csv_path}')

    json_path = os.path.join(out_dir, base + '_connections.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([{
            'signal': c['signal'],
            'from_device': c['from_device'],
            'to_device': c['to_device'],
        } for c in connections], f, indent=2, ensure_ascii=False)
    print(f'JSON: {json_path}')


def process(image_path, out_dir='./results'):
    """
    主处理流程

    Args:
        image_path: 输入图片路径
        out_dir: 输出目录

    Returns:
        list: 连接关系列表
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    print('\n=== Step 1: HAWP 线段检测 ===')
    detector = HawpDetector()
    hawp_lines, ori_w, ori_h = detector.detect_lines(image_path)

    print('\n=== Step 2: YOLO + OCR ===')
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    yolo_ocr_objects = process_single_page_image(img_b64)
    device_boxes, ground_boxes, power_boxes, signal_boxes = parse_objects(yolo_ocr_objects)

    print('\n=== Step 2.5: 为 signalName 添加辅助线段 ===')
    hawp_lines = add_auxiliary_lines_for_signal_boxes(hawp_lines, signal_boxes, device_boxes, expand_distance=50)

    print('\n=== Step 3: 构建图 ===')
    junctions, adj = GraphBuilder.build_graph(hawp_lines, Config.MERGE_TH)

    print('\n=== Step 4: 找设备入口节点 ===')
    device_entries, ground_entries, power_entries, entry_to_lines = GraphBuilder.find_entry_points(
        junctions, device_boxes, ground_boxes, power_boxes, hawp_lines
    )

    print('\n=== Step 4.5: 重建图（包含入口点）===')
    adj = GraphBuilder.rebuild_graph_with_entries(
        junctions, adj, entry_to_lines, hawp_lines, Config.MERGE_TH
    )

    print('\n=== Step 4.6: 合并 signalName box 内的 junction ===')
    junctions, adj = merge_junctions_in_signal_box(junctions, adj, hawp_lines, signal_boxes, device_boxes,
                                                   expand_distance=180, merge_threshold=5.0)

    print('\n=== Step 5: 搜索连接关系 ===')
    connections = ConnectionFinder.find_connections(
        junctions, adj, device_boxes, device_entries, signal_boxes,
        Config.OCR_DIST, ground_entries, ground_boxes, power_entries, power_boxes
    )

    print('\n--- 连接关系预览 ---')
    for c in connections:
        sig = f'[{c["signal"]}]' if c['signal'] else '[未知信号]'
        print(f'  {c["from_device"]} ←{sig}→ {c["to_device"]}')

    print('\n=== Step 6: 保存结果 ===')
    Visualizer.visualize(
        image_path, junctions, adj, device_boxes, device_entries,
        connections, os.path.join(out_dir, base + '_result.jpg'),
        ground_boxes, power_boxes, ground_entries, power_entries
    )
    save_results(connections, out_dir, base)

    return connections

if __name__ == '__main__':
    # 配置参数
    IMAGE_PATH = r'F:\office\pythonProjects\SystemVision-原理图识别\yolo\images\page_32_original.jpg'
    OUTPUT_DIR = './output'

    process(IMAGE_PATH, OUTPUT_DIR)
