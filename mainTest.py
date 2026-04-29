"""
主程序：HAWP线段检测 + BFS图搜索 + OCR信号名 → 连接关系表
"""
import os
import sys
import json
import csv
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

    # 执行名字匹配
    device_boxes = BoxUtils.match_device_names(device_boxes, signal_boxes)

    for dev in device_boxes:
        print(f'[DEBUG] Device at {dev["box"]} assigned name: "{dev["raw_text"]}"')

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

    print('\n=== Step 2: 构建图 ===')
    junctions, adj = GraphBuilder.build_graph(hawp_lines, Config.MERGE_TH)

    print('\n=== Step 3: YOLO + OCR ===')
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    yolo_ocr_objects = process_single_page_image(img_b64)
    device_boxes, ground_boxes, power_boxes, signal_boxes = parse_objects(yolo_ocr_objects)

    print('\n=== Step 4: 找设备入口节点 ===')
    device_entries, ground_entries, power_entries, entry_to_lines = GraphBuilder.find_entry_points(
        junctions, device_boxes, ground_boxes, power_boxes, hawp_lines
    )

    print('\n=== Step 4.5: 重建图（包含入口点）===')
    adj = GraphBuilder.rebuild_graph_with_entries(
        junctions, adj, entry_to_lines, hawp_lines, Config.MERGE_TH
    )

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
    IMAGE_PATH = r'F:\office\pythonProjects\SystemVision-原理图识别\yolo\images\page_28_original.jpg'
    OUTPUT_DIR = './output'

    process(IMAGE_PATH, OUTPUT_DIR)
