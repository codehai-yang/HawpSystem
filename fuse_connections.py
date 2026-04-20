"""
fuse_connections.py
把你已有的 pdf_service.py 的 YOLO+OCR 结果，
和 HAWP 的线段检测结果融合，提取引脚连接关系。

核心思路：
  1. YOLO 检测出所有 signalName 框（每个框里有引脚名称文字）
  2. HAWP 检测出所有线段（每条线有两个端点）
  3. 判断每个线段端点落在哪个 signalName 框里
  4. 两个端点分别属于不同的 signalName → 这两个引脚之间有连接

用法：
  from fuse_connections import fuse_and_extract
  connections = fuse_and_extract(image_path, yolo_ocr_results, hawp_lines)
"""

import os
import cv2
import math
import json
import csv
import numpy as np
from collections import defaultdict
from app.services.pdf_service import process_single_page_image
from UseModel import load_model, predict_large_image




# ═══════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════

def point_in_box(px, py, box, padding=6):
    """判断点 (px,py) 是否在 box 内，box=[x_min,y_min,x_max,y_max]，padding 扩展容差"""
    x_min, y_min, x_max, y_max = box
    return (x_min - padding <= px <= x_max + padding and
            y_min - padding <= py <= y_max + padding)


def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def box_center(box):
    x_min, y_min, x_max, y_max = box
    return ((x_min+x_max)/2, (y_min+y_max)/2)


def nearest_box(px, py, boxes):
    """找到距离点最近的 box，返回 (index, distance)"""
    best_idx, best_d = None, float('inf')
    for i, box in enumerate(boxes):
        cx, cy = box_center(box)
        d = dist((px, py), (cx, cy))
        if d < best_d:
            best_d = d
            best_idx = i
    return best_idx, best_d


# ═══════════════════════════════════════════
#  核心融合逻辑
# ═══════════════════════════════════════════

def fuse_and_extract(image_path,
                     yolo_ocr_objects,
                     hawp_lines,
                     endpoint_box_padding=8,
                     near_box_threshold=30,
                     merge_threshold=10):
    """
    融合 YOLO+OCR 结果和 HAWP 线段，提取连接关系。

    参数：
        image_path         : 原始图片路径（用于可视化）
        yolo_ocr_objects   : pdf_service.py 返回的 objects 列表
                             格式：[{'label':'signalName','points':[{x,y}x4],'text':'KL30'}, ...]
        hawp_lines         : HAWP 检测的线段列表
                             格式：[{'p1':(x,y),'p2':(x,y),'score':float}, ...]
        endpoint_box_padding: 端点匹配框时的容差像素
        near_box_threshold : 端点不在框内时，距离小于此值也算匹配（单位：像素）
        merge_threshold    : 相同引脚名称的重复端点合并距离

    返回：
        connections : [{'from': '引脚A名', 'to': '引脚B名', ...}, ...]
    """

    # ── 1. 整理 signalName 框 ──
    signal_boxes = []  # [{'text':str, 'box':[x_min,y_min,x_max,y_max], 'points':...}, ...]
    for obj in yolo_ocr_objects:
        if obj.get('label') != 'signalName':
            continue
        pts = obj.get('points', [])
        if len(pts) < 4:
            continue
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        signal_boxes.append({
            'text':  obj.get('text', '').strip(),
            'box':   [min(xs), min(ys), max(xs), max(ys)],
            'points': pts
        })

    print(f'signalName 框数量: {len(signal_boxes)}')
    # ── 1. 整理 deviceName 框 ──
    device_boxes = []  # [{'text':str, 'box':[x_min,y_min,x_max,y_max], 'points':...}, ...]
    for obj in yolo_ocr_objects:
        if obj.get('label') != 'deviceName':
            continue
        pts = obj.get('points', [])
        if len(pts) < 4:
            continue
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        device_boxes.append({
            'text':  obj.get('text', '').strip(),
            'box':   [min(xs), min(ys), max(xs), max(ys)],
            'points': pts
        })

    print(f'deviceName 框数量: {len(device_boxes)}')

    # ── 2. 每条线段的两个端点匹配到 signalName 框 ──
    def match_endpoint(px, py):
        """
        尝试把端点匹配到 signalName 框，返回 (box_index, method)
        method: 'in_box' | 'near_box' | None
        """
        # 优先：端点在框内
        for i, sb in enumerate(signal_boxes):
            if point_in_box(px, py, sb['box'], endpoint_box_padding):
                return i, 'in_box'
        # 次优：端点到框中心距离小于阈值
        best_idx, best_d = nearest_box(px, py, [sb['box'] for sb in signal_boxes])
        if best_idx is not None and best_d < near_box_threshold:
            return best_idx, 'near_box'
        return None, None

    # ── 3. 遍历所有线段，找两端都能匹配到 signalName 的线段 ──
    raw_connections = []
    matched_count   = 0

    for line in hawp_lines:
        p1x, p1y = line['p1']
        p2x, p2y = line['p2']

        idx1, method1 = match_endpoint(p1x, p1y)
        idx2, method2 = match_endpoint(p2x, p2y)

        if idx1 is None or idx2 is None:
            continue
        if idx1 == idx2:
            continue  # 两端匹配到同一个框，跳过

        matched_count += 1
        name1 = signal_boxes[idx1]['text']
        name2 = signal_boxes[idx2]['text']

        raw_connections.append({
            'from':       name1,
            'to':         name2,
            'from_box':   idx1,
            'to_box':     idx2,
            'from_xy':    (p1x, p1y),
            'to_xy':      (p2x, p2y),
            'score':      line.get('score', 1.0),
            'method':     f'{method1}+{method2}'
        })

    print(f'两端都匹配到 signalName 的线段: {matched_count} 条')

    # ── 4. 去重（同一对引脚可能有多条线段匹配到，保留置信度最高的）──
    pair_best = {}
    for conn in raw_connections:
        pair = tuple(sorted([conn['from'], conn['to']]))
        if pair not in pair_best or conn['score'] > pair_best[pair]['score']:
            pair_best[pair] = conn

    connections = list(pair_best.values())
    print(f'去重后连接关系: {len(connections)} 对')

    return connections, signal_boxes,device_boxes


# ═══════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════

def visualize_connections(image_path, signal_boxes,device_boxes, hawp_lines,
                          connections, out_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f'找不到图片: {image_path}')
        return

    # 画所有 HAWP 线段（灰色细线）
    for line in hawp_lines:
        x1, y1 = int(line['p1'][0]), int(line['p1'][1])
        x2, y2 = int(line['p2'][0]), int(line['p2'][1])
        cv2.line(image, (x1,y1), (x2,y2), (200,0,200), 1)
    # 画 deviceName 框（红色）
    for db in device_boxes:
        x_min,y_min,x_max,y_max = [int(v) for v in db['box']]
        cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0, 0, 200), 1)
        cv2.putText(image, db['text'],
                    (x_min, y_min-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 180), 1)
    # 画 signalName 框（蓝色）
    for sb in signal_boxes:
        x_min,y_min,x_max,y_max = [int(v) for v in sb['box']]
        cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (200,120,0), 1)
        cv2.putText(image, sb['text'],
                    (x_min, y_min-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (180,100,0), 1)

    # 画连接关系（绿色连线，连两个框的中心）
    for conn in connections:
        sb1 = signal_boxes[conn['from_box']]
        sb2 = signal_boxes[conn['to_box']]
        c1  = box_center(sb1['box'])
        c2  = box_center(sb2['box'])
        cv2.line(image,
                 (int(c1[0]), int(c1[1])),
                 (int(c2[0]), int(c2[1])),
                 (0, 180, 60), 2)
        # 连线中点标注引脚名
        mx = int((c1[0]+c2[0])/2)
        my = int((c1[1]+c2[1])/2)
        label = f'{conn["from"]} ↔ {conn["to"]}'
        cv2.putText(image, label, (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0,140,40), 1)

    cv2.putText(image, f'connections: {len(connections)}',
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,180), 2)

    cv2.imwrite(out_path, image)
    print(f'可视化保存: {out_path}')


# ═══════════════════════════════════════════
#  保存结果
# ═══════════════════════════════════════════

def save_csv(connections, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['引脚A', '引脚B', '置信度'])
        for conn in connections:
            writer.writerow([conn['from'], conn['to'],
                             round(conn['score'], 3)])
    print(f'连接关系表: {out_path}')


def save_json(connections, out_path):
    data = [{'from': c['from'], 'to': c['to'],
             'score': round(c['score'], 3)} for c in connections]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'JSON结果: {out_path}')


# ═══════════════════════════════════════════
#  完整流程入口（把你的 pdf_service 和 HAWP 接在一起）
# ═══════════════════════════════════════════

def process_schematic(image_path,
                      out_dir='./results',
                      hawp_ckpt=None,
                      hawp_cfg=None,
                      device='cuda'):
    """
    完整流程：图片 → YOLO+OCR → HAWP → 融合 → 连接关系表

    如果你已经有 yolo_ocr_objects 和 hawp_lines，
    可以直接调用 fuse_and_extract() 跳过检测步骤。
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ── Step 1: YOLO + OCR（复用你已有的 pdf_service）──
    print('\n=== Step 1: YOLO + OCR ===')
    import base64 as b64
    with open(image_path, 'rb') as f:
        img_b64 = b64.b64encode(f.read()).decode()

    yolo_ocr_objects = process_single_page_image(img_b64)
    print(f'YOLO+OCR 返回 {len(yolo_ocr_objects)} 个对象')

    # ── Step 2: HAWP 线段检测 ──
    print('\n=== Step 2: HAWP 线段检测 ===')
    hawp_model = load_model(hawp_ckpt, hawp_cfg, device)
    hawp_lines, _, _ = predict_large_image(
        hawp_model, image_path, device,
        threshold=0.25, infer_size=512,
        tile_size=1024, overlap=128)

    # ── Step 3: 融合 ──
    print('\n=== Step 3: 融合提取连接关系 ===')
    connections, signal_boxes,device_boxes = fuse_and_extract(
        image_path, yolo_ocr_objects, hawp_lines)

    # 打印预览
    print('\n--- 连接关系（前20条）---')
    for conn in connections[:20]:
        print(f'  {conn["from"]}  ←→  {conn["to"]}  (score={conn["score"]:.2f})')

    # ── Step 4: 保存 ──
    visualize_connections(
        image_path, signal_boxes,device_boxes, hawp_lines, connections,
        os.path.join(out_dir, base+'_connections.jpg'))
    save_csv(connections,
             os.path.join(out_dir, base+'_connections.csv'))
    save_json(connections,
              os.path.join(out_dir, base+'_connections.json'))

    return connections


# ═══════════════════════════════════════════
#  如果你想单独测试融合逻辑（不跑完整流程）
# ═══════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',       required=True)
    parser.add_argument('--out_dir',   default='./results')
    parser.add_argument('--hawp_ckpt', default=r'F:\office\pythonProjects\YOLOandOCR\AutoLogic\BackendOCR\best.pth')
    parser.add_argument('--hawp_cfg',  default=r'F:\office\pythonProjects\YOLOandOCR\AutoLogic\BackendOCR\hawp\fsl\config\hawpv2.yaml')
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    process_schematic(
        args.img, args.out_dir,
        args.hawp_ckpt, args.hawp_cfg, args.device)