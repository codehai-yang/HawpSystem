"""
hawp_bfs_connections.py
完整流程：HAWP线段检测 + BFS图搜索 + OCR信号名 → 连接关系表

用法：
  python hawp_bfs_connections.py --img your_image.jpg --out_dir ./results
"""

import os
import sys
import cv2
import math
import json
import csv
import base64
import torch
import numpy as np
from collections import defaultdict, deque
from app.services.pdf_service import process_single_page_image


# ── HAWP 路径，改成你的实际路径 ──────────────────────
HAWP_ROOT = r'F:\office\pythonProjects\HAWP\hawp'
if HAWP_ROOT not in sys.path:
    sys.path.insert(0, HAWP_ROOT)
# ─────────────────────────────────────────────────────

# ── 配置区 ────────────────────────────────────────────
CKPT_PATH      = r'F:\office\pythonProjects\YOLOandOCR\AutoLogic\BackendOCR\hawpv2-edb9b23f.pth'
CFG_PATH       = r'F:\office\pythonProjects\YOLOandOCR\AutoLogic\BackendOCR\hawpv2.yaml'
DEVICE         = 'cuda'
HAWP_THRESHOLD = 0.25
INFER_SIZE     = 512
TILE_SIZE      = 1024
OVERLAP        = 128
MERGE_TH       = 15      # 端点合并距离（像素），分辨率高可调大
EDGE_TOL       = 25      # 设备框边缘容差（像素）
OCR_DIST       = 50      # OCR文字匹配距离（像素）
# ─────────────────────────────────────────────────────


# ═══════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def point_near_box_edge(px, py, box, tol):
    """点在设备框边缘 tol 像素范围内"""
    x1,y1,x2,y2 = box
    in_outer = (x1-tol <= px <= x2+tol) and (y1-tol <= py <= y2+tol)
    in_inner = (x1+tol <= px <= x2-tol) and (y1+tol <= py <= y2-tol)
    return in_outer and not in_inner

def point_in_box(px, py, box, pad=0):
    x1,y1,x2,y2 = box
    return x1-pad <= px <= x2+pad and y1-pad <= py <= y2+pad


# ═══════════════════════════════════════════
#  Step 1：HAWP 推理
# ═══════════════════════════════════════════

def load_hawp():
    from hawp.fsl.config import cfg as model_config
    from hawp.fsl.model.build import build_model
    model_config.merge_from_file(CFG_PATH)
    model = build_model(model_config)
    model = model.eval().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location='cpu')
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    print(f'HAWP 加载成功: {os.path.basename(CKPT_PATH)}')
    return model


def predict_tile(model, tile_bgr):
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    h, w = tile_rgb.shape[:2]
    resized = cv2.resize(tile_rgb, (INFER_SIZE, INFER_SIZE))
    tensor  = torch.from_numpy(resized).float() / 255.0
    tensor  = tensor.permute(2,0,1)[None].to(DEVICE)
    meta    = [{'width':INFER_SIZE,'height':INFER_SIZE,'filename':''}]
    try:
        with torch.no_grad():
            output, _ = model(tensor, meta)
    except Exception:
        return []
    lines  = output['lines_pred'].cpu().numpy()
    scores = output['lines_score'].cpu().numpy().flatten()
    sx, sy = w/INFER_SIZE, h/INFER_SIZE
    results = []
    for line, score in zip(lines, scores):
        if score < HAWP_THRESHOLD:
            continue
        results.append({
            'p1': (float(line[0])*sx, float(line[1])*sy),
            'p2': (float(line[2])*sx, float(line[3])*sy),
            'score': float(score)
        })
    return results


def hawp_predict(model, image_path):
    image = cv2.imread(image_path)
    ori_h, ori_w = image.shape[:2]
    # 步长
    stride = TILE_SIZE - OVERLAP
    #计算需要的列数和行数
    n_cols = math.ceil((ori_w - OVERLAP) / stride)
    n_rows = math.ceil((ori_h - OVERLAP) / stride)
    # 计算总瓦片数量
    total  = n_rows * n_cols
    # 存储所有检测到的线条
    all_lines = []
    # 计数器，跟踪已处理的瓦片数
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            count += 1
            x1 = col * stride
            y1 = row * stride
            x2 = min(x1+TILE_SIZE, ori_w)
            y2 = min(y1+TILE_SIZE, ori_h)
            tile = image[y1:y2, x1:x2]
            for line in predict_tile(model, tile):
                all_lines.append({
                    'p1':   (line['p1'][0]+x1, line['p1'][1]+y1),
                    'p2':   (line['p2'][0]+x1, line['p2'][1]+y1),
                    'score': line['score']
                })
            print(f'  HAWP [{count}/{total}]', end='\r')
    print(f'\nHAWP 检测到 {len(all_lines)} 条线段')
    return all_lines, ori_w, ori_h


# ═══════════════════════════════════════════
#  Step 2：端点合并 → 构建图
# ═══════════════════════════════════════════

def build_graph(hawp_lines, merge_threshold):
    """
    端点合并（Union-Find）+ 构建邻接表。
    返回 junctions 列表和 adj 邻接表。
    """
    raw_pts = []
    for seg in hawp_lines:
        raw_pts.append(seg['p1'])
        raw_pts.append(seg['p2'])

    # HAWP检测出的所有端点数量
    n = len(raw_pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # 距离小于阈值的端点合并
    for i in range(n):
        for j in range(i+1, n):
            # 如果两个端点距离小于阈值，合并
            if dist(raw_pts[i], raw_pts[j]) < merge_threshold:
                union(i, j)

    # 整理分组，坐标取均值
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    root_to_jid = {}
    junctions = []
    for root, members in groups.items():
        jid = len(junctions)
        root_to_jid[root] = jid
        cx = sum(raw_pts[m][0] for m in members) / len(members)
        cy = sum(raw_pts[m][1] for m in members) / len(members)
        junctions.append({'id': jid, 'x': cx, 'y': cy})

    # 构建邻接表
    adj = defaultdict(set)
    for idx, seg in enumerate(hawp_lines):
        pi, pj = idx*2, idx*2+1
        ja = root_to_jid[find(pi)]
        jb = root_to_jid[find(pj)]
        if ja != jb:
            adj[ja].add(jb)
            adj[jb].add(ja)

    print(f'构建图：{len(junctions)} 个节点，'
          f'{sum(len(v) for v in adj.values())//2} 条边')
    return junctions, dict(adj)


# ═══════════════════════════════════════════
#  Step 3：YOLO+OCR 解析
# ═══════════════════════════════════════════

def match_device_names(device_boxes, signal_boxes, corner_margin=30, top_search_range=50, center_align_threshold=20):
    """
    为每个 device 匹配一个最合适的 signalName 作为 raw_text。

    搜索策略（按优先级）：
    1. 优先查找设备框左上角和右上角内部的 signalName（必须在框内）
    2. 如果没找到，从顶部边中点向上查找一定距离（框外）
    3. 如果还没找到，从顶部边中点向下查找（框内靠近顶部）
    4. 最后，在设备框内部查找与顶部中点水平对齐的信号名（中心点对齐）

    参数：
    - corner_margin: 角落搜索区域的范围（默认30像素）
    - top_search_range: 顶部中点向上/向下搜索的范围（默认50像素）
    - center_align_threshold: 中心点对齐的允许偏移量（默认20像素）
    """
    for dev in device_boxes:
        dx1, dy1, dx2, dy2 = dev['box']
        dev_width = dx2 - dx1

        # 定义关键位置
        top_center_x = (dx1 + dx2) / 2  # 顶部中点x坐标

        best_signal = None
        best_priority = 999  # 优先级：1=角落(框内), 2=上方, 3=下方, 4=框内居中对齐
        min_dist = float('inf')

        for sig in signal_boxes:
            sx1, sy1, sx2, sy2 = sig['box']
            sig_center_x = (sx1 + sx2) / 2
            sig_center_y = (sy1 + sy2) / 2

            # 判断信号名的位置区域

            # 优先级1：左上角（必须在设备框内部）
            is_left_corner = (sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2 and
                              (sx2 - dx1) < corner_margin and abs(sig_center_y - dy1) < corner_margin * 1.5)

            # 优先级1：右上角（必须在设备框内部）
            is_right_corner = (sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2 and
                               (dx2 - sx1) < corner_margin and abs(sig_center_y - dy1) < corner_margin * 1.5)

            # 优先级2：顶部中点上方（在设备框外部）
            is_above_center = (sy2 < dy1) and (dx1 - 20 <= sig_center_x <= dx2 + 20) and (dy1 - top_search_range <= sig_center_y <= dy1)

            # 优先级3：顶部中点下方（在设备框内部靠近顶部）
            is_below_center = (sy1 > dy1) and (dx1 - 20 <= sig_center_x <= dx2 + 20) and (dy1 <= sig_center_y <= dy1 + top_search_range)

            # 优先级4：设备框内部，中心点与顶部中点水平对齐
            is_inside_center_aligned = (sx1 >= dx1 and sx2 <= dx2 and sy1 >= dy1 and sy2 <= dy2 and
                                        abs(sig_center_x - top_center_x) <= center_align_threshold)

            # 确定优先级和距离
            priority = 999
            dist_to_ref = float('inf')

            if is_left_corner or is_right_corner:
                # 优先级1：角落（框内）
                priority = 1
                if is_left_corner:
                    dist_to_ref = math.sqrt((sig_center_x - dx1)**2 + (sig_center_y - dy1)**2)
                else:
                    dist_to_ref = math.sqrt((sig_center_x - dx2)**2 + (sig_center_y - dy1)**2)
            elif is_above_center:
                # 优先级2：顶部中点上方
                priority = 2
                dist_to_ref = abs(sig_center_y - dy1)
            elif is_below_center:
                # 优先级3：顶部中点下方
                priority = 3
                dist_to_ref = abs(sig_center_y - dy1)
            elif is_inside_center_aligned:
                # 优先级4：框内居中对齐
                priority = 4
                dist_to_ref = abs(sig_center_y - dy1)  # 距离顶部边的垂直距离

            # 如果不在任何搜索区域内，跳过
            if priority == 999:
                continue

            # 比较优先级和距离
            if priority < best_priority:
                # 更高优先级，直接选中
                best_signal = sig
                best_priority = priority
                min_dist = dist_to_ref
            elif priority == best_priority:
                # 相同优先级，选择距离更近的
                if dist_to_ref < min_dist:
                    best_signal = sig
                    min_dist = dist_to_ref

        # 更新 raw_text
        if best_signal:
            dev['raw_text'] = best_signal['raw_text']

            # 确定位置信息用于调试
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
            else:
                position_info = f"框内居中(偏移{abs(best_signal['center'][0] - top_center_x):.1f}px)"

            print(f'[DEBUG] Device at {dev["box"]} → "{dev["raw_text"]}" (位置: {position_info}, 优先级: {best_priority}, 距离: {min_dist:.1f})')
        else:
            dev['raw_text'] = "Unknown"
            print(f'[DEBUG] Device at {dev["box"]} → "Unknown" (未找到信号名)')

    return device_boxes


def parse_objects(objects):
    """
    解析 PDF Service 返回的 objects，分离 device、ground、power 和 signalName，并执行名字匹配
    """
    device_boxes = []
    ground_boxes = []
    power_boxes = []
    signal_boxes = []

    for idx, obj in enumerate(objects):
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

        # 只有 signalName 才提取 text，其他类别暂时留空
        label = obj.get('label', '')
        text = obj.get('text', '').strip() if label == 'signalName' else ''

        entry = {
            'raw_text': text,
            'box':      box,
            'center':   box_center(box),
            'label':    label,
        }

        if label == 'device':
            device_boxes.append(entry)
        elif label == 'ground':
            ground_boxes.append(entry)
        elif label == 'power':
            power_boxes.append(entry)
        elif label == 'signalName':
            signal_boxes.append(entry)

    print(f'\nParsed: {len(device_boxes)} devices, {len(ground_boxes)} grounds, {len(power_boxes)} powers, {len(signal_boxes)} signals')
    # 合并分裂的 device 框
    device_boxes = merge_split_device_boxes(device_boxes)
    # 执行名字匹配逻辑，直接修改 device_boxes 中的 raw_text
    device_boxes = match_device_names(device_boxes, signal_boxes)

    # 调试打印
    for dev in device_boxes:
        print(f'[DEBUG] Device at {dev["box"]} assigned name: "{dev["raw_text"]}"')

    return device_boxes, ground_boxes, power_boxes, signal_boxes

# ═══════════════════════════════════════════
#  Step 4：找设备入口节点
# ═══════════════════════════════════════════

def find_entry_points(junctions, device_boxes, edge_tol, ground_boxes, power_boxes, hawp_lines):
    """
    找每个设备框、ground框、power框的入口节点。
    新逻辑：只要HAWP检测的线段与设备框有交点，交点就是入口点（直接创建新的junction）。
    返回 {dev_idx: [jid, ...]}, {ground_idx: [jid, ...]}, {power_idx: [jid, ...]}
    """
    device_entries = defaultdict(list)
    ground_entries = defaultdict(list)
    power_entries = defaultdict(list)

    # 用于存储所有新创建的入口点
    new_junctions = []
    next_jid = len(junctions)

    # 记录每个入口点对应的HAWP线段索引，用于后续建图
    entry_to_lines = []  # [(entry_jid, line_index), ...]

    # 为每个设备类型收集所有相关的交点
    def collect_intersections(boxes, entries_dict):
        nonlocal next_jid

        for box_idx, box_info in enumerate(boxes):
            box = box_info['box']
            entry_jids = []

            # 遍历所有HAWP线段，找与当前框相交的线段
            for line_idx, line in enumerate(hawp_lines):
                p1 = line['p1']
                p2 = line['p2']

                # 计算线段与框的交点
                intersections = line_box_intersections(p1, p2, box)

                # 对于每个交点，创建一个新的junction
                for ix, iy in intersections:
                    jid = next_jid
                    next_jid += 1
                    new_junction = {'id': jid, 'x': ix, 'y': iy}
                    new_junctions.append(new_junction)
                    entry_jids.append(jid)
                    entry_to_lines.append((jid, line_idx))

            if entry_jids:
                entries_dict[box_idx] = entry_jids

    # 收集所有设备类型的入口点
    collect_intersections(device_boxes, device_entries)
    collect_intersections(ground_boxes, ground_entries)
    collect_intersections(power_boxes, power_entries)

    # 将新创建的入口点添加到junctions列表中
    junctions.extend(new_junctions)

    print(f'\n新增入口点数量: {len(new_junctions)}')
    print(f'总junction数量: {len(junctions)}')

    print('\n设备入口点:')
    for dev_idx, jids in device_entries.items():
        print(f'  [{device_boxes[dev_idx]["raw_text"]}]: {len(jids)} 个入口节点')

    print('\nGround 入口点:')
    for ground_idx, jids in ground_entries.items():
        box = ground_boxes[ground_idx]['box']
        print(f'  [Ground at {box}]: {len(jids)} 个入口节点')

    print('\nPower 入口点:')
    for power_idx, jids in power_entries.items():
        box = power_boxes[power_idx]['box']
        print(f'  [Power at {box}]: {len(jids)} 个入口节点')

    return dict(device_entries), dict(ground_entries), dict(power_entries), entry_to_lines


def rebuild_graph_with_entries(original_junctions, original_adj, new_entries, entry_to_lines, hawp_lines, merge_threshold):
    """
    重新构建图，将入口点连接到对应的HAWP线段端点

    参数：
    - original_junctions: 原始的junctions列表
    - original_adj: 原始的邻接表
    - new_entries: 新创建的入口点列表
    - entry_to_lines: 入口点到HAWP线段的映射 [(entry_jid, line_idx), ...]
    - hawp_lines: HAWP检测的线段列表
    - merge_threshold: 合并阈值

    返回：
    - 更新后的邻接表
    """
    from collections import defaultdict

    # 创建新的邻接表（复制原始的）
    new_adj = defaultdict(set)
    for jid, neighbors in original_adj.items():
        for nb in neighbors:
            new_adj[jid].add(nb)
            new_adj[nb].add(jid)

    # 为每个入口点找到它所属的HAWP线段的两个端点对应的junction
    # 首先建立HAWP线段端点到junction的映射
    line_to_endpoints = {}  # line_idx -> (endpoint_jid_1, endpoint_jid_2)

    # 重建线段端点到junction的映射
    raw_pts = []
    for seg in hawp_lines:
        raw_pts.append(seg['p1'])
        raw_pts.append(seg['p2'])

    n = len(raw_pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # 距离小于阈值的端点合并
    for i in range(n):
        for j in range(i+1, n):
            if dist(raw_pts[i], raw_pts[j]) < merge_threshold:
                union(i, j)

    # 整理分组
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    root_to_jid = {}
    junction_id_map = []
    jid_counter = 0
    for root, members in groups.items():
        root_to_jid[root] = jid_counter
        junction_id_map.append(members)
        jid_counter += 1

    # 建立线段索引到端点junction的映射
    for line_idx in range(len(hawp_lines)):
        pi, pj = line_idx * 2, line_idx * 2 + 1
        ja = root_to_jid[find(pi)]
        jb = root_to_jid[find(pj)]
        line_to_endpoints[line_idx] = (ja, jb)

    # 为每个入口点添加连接到对应线段的两个端点
    for entry_jid, line_idx in entry_to_lines:
        if line_idx in line_to_endpoints:
            ep1, ep2 = line_to_endpoints[line_idx]
            # 将入口点连接到线段的两个端点
            new_adj[entry_jid].add(ep1)
            new_adj[ep1].add(entry_jid)
            new_adj[entry_jid].add(ep2)
            new_adj[ep2].add(entry_jid)

    print(f'[重建图] 添加了 {len(entry_to_lines)} 个入口点连接')
    print(f'[重建图] 总边数: {sum(len(v) for v in new_adj.values()) // 2}')

    return dict(new_adj)

def line_segment_intersection(p1, p2, p3, p4):
    """
    计算两条线段的交点
    返回交点坐标 (x, y) 或 None（如果不相交）
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # 计算分母
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # 平行线

    # 计算交点参数
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # 检查交点是否在线段范围内
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None

def line_box_intersections(line_start, line_end, box):
    """
    计算线段与矩形框的所有交点
    返回交点列表 [(x, y), ...]
    """
    x1, y1, x2, y2 = box
    intersections = []

    # 矩形的四条边
    edges = [
        ((x1, y1), (x2, y1)),  # 上边
        ((x2, y1), (x2, y2)),  # 右边
        ((x2, y2), (x1, y2)),  # 下边
        ((x1, y2), (x1, y1))   # 左边
    ]

    for edge_start, edge_end in edges:
        point = line_segment_intersection(line_start, line_end, edge_start, edge_end)
        if point is not None:
            intersections.append(point)

    return intersections

# ═══════════════════════════════════════════
#  Step 5：BFS 搜索连通路径
# ═══════════════════════════════════════════

def bfs_path_exists(start_jid, target_set, adj,
                    exclude_jids=None, max_depth=200):
    """
    从 start_jid 出发 BFS，找能否到达 target_set 里的任意节点。
    exclude_jids: 不能经过的节点集合（避免穿越其他设备内部）
    返回：(能否到达, 到达的目标节点id)
    """
    if exclude_jids is None:
        exclude_jids = set()

    visited = {start_jid}
    queue   = deque([(start_jid, 0)])

    while queue:
        cur, depth = queue.popleft()
        if depth > max_depth:
            continue
        for nb in adj.get(cur, []):
            if nb in visited:
                continue
            if nb in exclude_jids:
                continue
            if nb in target_set:
                return True, nb
            visited.add(nb)
            queue.append((nb, depth+1))

    return False, None


def find_connections_bfs(junctions, adj, device_boxes, device_entries, signal_boxes,
                         ocr_dist, ground_entries, ground_boxes, power_entries, power_boxes):
    """
    对每对设备进行 BFS 搜索，找到所有可能的连接路径。

    优化策略：
    1. 根据设备相对方位，只搜索相关方向的入口点
    2. 找到所有可达路径，而不是只找第一条
    3. 在每条路径上查找与路径相交或在上方的 signalName
    """
    # id对应的x,y坐标字典
    jmap = {j['id']: j for j in junctions}

    # 合并所有设备类型为一个列表，方便遍历
    all_devices = []
    all_entries = {}
    offset = 0

    # 添加 device
    for idx, dev in enumerate(device_boxes):
        dev['type'] = 'device'
        all_devices.append(dev)
        if idx in device_entries:
            all_entries[idx + offset] = device_entries[idx]
    device_count = len(device_boxes)
    offset += device_count

    # 添加 ground
    for idx, ground in enumerate(ground_boxes):
        ground['type'] = 'ground'
        all_devices.append(ground)
        if idx in ground_entries:
            all_entries[idx + offset] = ground_entries[idx]
    ground_count = len(ground_boxes)
    offset += ground_count

    # 添加 power
    for idx, power in enumerate(power_boxes):
        power['type'] = 'power'
        all_devices.append(power)
        if idx in power_entries:
            all_entries[idx + offset] = power_entries[idx]

    n_devs = len(all_devices)
    connections = []

    # 预计算：哪些 junction 在设备框内部
    interior_jids = set()
    for j in junctions:
        for dev in all_devices:
            if point_in_box(j['x'], j['y'], dev['box'], pad=-5):
                interior_jids.add(j['id'])

    print(f'\n[BFS] 开始搜索连接关系，共 {n_devs} 个设备（{device_count} device, {ground_count} ground, {len(power_boxes)} power）')

    for dev_a in range(n_devs):
        entries_a = all_entries.get(dev_a, [])
        if not entries_a:
            continue

        dev_a_info = all_devices[dev_a]

        for dev_b in range(dev_a + 1, n_devs):
            entries_b = all_entries.get(dev_b, [])
            if not entries_b:
                continue

            dev_b_info = all_devices[dev_b]

            # 根据相对方位筛选入口点
            filtered_entries_a, filtered_entries_b = filter_entries_by_direction(
                dev_a_info, dev_b_info, entries_a, entries_b, jmap
            )

            if not filtered_entries_a or not filtered_entries_b:
                continue

            target_set = set(filtered_entries_b)

            # 找到所有可达路径
            all_paths = find_all_paths(
                filtered_entries_a,
                target_set,
                adj,
                interior_jids - target_set,
                max_depth=30
            )

            # 为每条路径创建连接记录
            for path_jids in all_paths:
                if len(path_jids) < 2:
                    continue

                entry_jid = path_jids[0]
                reached_jid = path_jids[-1]

                # 在路径上查找信号名
                signal = find_signal_on_path(path_jids, jmap, signal_boxes, ocr_dist)

                connections.append({
                    'from_device': dev_a_info.get('raw_text', 'Unknown'),
                    'to_device': dev_b_info.get('raw_text', 'Unknown'),
                    'from_type': dev_a_info.get('type', 'device'),
                    'to_type': dev_b_info.get('type', 'device'),
                    'signal': signal,
                    'from_entry': entry_jid,
                    'to_entry': reached_jid,
                    'from_center': (jmap[entry_jid]['x'], jmap[entry_jid]['y']),
                    'to_center': (jmap[reached_jid]['x'], jmap[reached_jid]['y']),
                    'path_jids': path_jids
                })

    print(f'[BFS] 找到 {len(connections)} 条连接路径')
    return connections

def find_all_paths(start_jids, target_set, adj, exclude_jids, max_depth=30):
    """
    从多个起点出发，找到所有能到达目标集合的路径

    返回：
    - 所有路径的列表，每条路径是一个 junction id 列表
    """
    all_paths = []
    visited_global = set()  # 全局已访问，避免重复路径

    for start_jid in start_jids:
        # BFS 搜索所有路径
        visited = {start_jid}
        # queue 中存储 (当前节点, 当前路径)
        queue = deque([(start_jid, [start_jid])])

        while queue:
            cur, path = queue.popleft()

            # 如果达到最大深度，停止扩展
            if len(path) > max_depth:
                continue

            # 如果当前节点是目标节点，记录路径
            if cur in target_set:
                path_tuple = tuple(path)
                if path_tuple not in visited_global:
                    visited_global.add(path_tuple)
                    all_paths.append(path)
                # 继续搜索，不break，寻找其他路径
                continue

            # 扩展邻居节点
            for nb in adj.get(cur, []):
                if nb not in visited and nb not in exclude_jids:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))

    return all_paths

def get_relative_position(dev_a, dev_b):
    """
    判断 dev_b 相对于 dev_a 的方位

    返回：'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    center_a = dev_a['center']
    center_b = dev_b['center']

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    # 计算角度
    import math
    angle = math.degrees(math.atan2(dy, dx))

    # 将角度转换为8个方向
    if -22.5 <= angle < 22.5:
        return 'right'
    elif 22.5 <= angle < 67.5:
        return 'bottom_right'
    elif 67.5 <= angle < 112.5:
        return 'bottom'
    elif 112.5 <= angle < 157.5:
        return 'bottom_left'
    elif angle >= 157.5 or angle < -157.5:
        return 'left'
    elif -157.5 <= angle < -112.5:
        return 'top_left'
    elif -112.5 <= angle < -67.5:
        return 'top'
    else:
        return 'top_right'



def filter_entries_by_direction(dev_a, dev_b, entries_a, entries_b, jmap):
    """
    根据设备相对方位，筛选需要搜索的入口点
    """
    direction = get_relative_position(dev_a, dev_b)

    # 定义每个方向应该使用的边
    # 对于 dev_a，如果 dev_b 在右边，则 dev_a 只需要搜索右、上、下边
    direction_to_edges = {
        'right': ['right', 'top', 'bottom'],
        'left': ['left', 'top', 'bottom'],
        'top': ['top', 'left', 'right'],
        'bottom': ['bottom', 'left', 'right'],
        'top_right': ['right', 'top'],
        'top_left': ['left', 'top'],
        'bottom_right': ['right', 'bottom'],
        'bottom_left': ['left', 'bottom']
    }

    allowed_edges_a = direction_to_edges.get(direction, ['top', 'bottom', 'left', 'right'])

    # 反向：对于 dev_b，dev_a 在相反方向
    opposite_direction = {
        'right': 'left', 'left': 'right',
        'top': 'bottom', 'bottom': 'top',
        'top_right': 'bottom_left', 'bottom_left': 'top_right',
        'top_left': 'bottom_right', 'bottom_right': 'top_left'
    }

    allowed_edges_b = direction_to_edges.get(opposite_direction[direction], ['top', 'bottom', 'left', 'right'])

    # 筛选 dev_a 的入口点
    filtered_a = []
    for jid in entries_a:
        j = jmap[jid]
        edge = get_point_edge(j['x'], j['y'], dev_a['box'])
        if edge in allowed_edges_a:
            filtered_a.append(jid)

    # 筛选 dev_b 的入口点
    filtered_b = []
    for jid in entries_b:
        j = jmap[jid]
        edge = get_point_edge(j['x'], j['y'], dev_b['box'])
        if edge in allowed_edges_b:
            filtered_b.append(jid)

    return filtered_a, filtered_b

def get_point_edge(px, py, box):
    """判断点在矩形框的哪条边上"""
    x1, y1, x2, y2 = box
    tol = 25  # 容差

    dist_to_top = abs(py - y1)
    dist_to_bottom = abs(py - y2)
    dist_to_left = abs(px - x1)
    dist_to_right = abs(px - x2)

    min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)

    if min_dist == dist_to_top:
        return 'top'
    elif min_dist == dist_to_bottom:
        return 'bottom'
    elif min_dist == dist_to_left:
        return 'left'
    else:
        return 'right'

def reconstruct_path(start_jid, end_jid, adj, exclude_jids):
    """
    BFS 重构路径，返回从起点到终点的所有 junction id
    """
    from collections import deque

    visited = {start_jid}
    queue = deque([(start_jid, [start_jid])])

    while queue:
        cur, path = queue.popleft()

        if cur == end_jid:
            return path

        for nb in adj.get(cur, []):
            if nb not in visited and nb not in exclude_jids:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return [start_jid, end_jid]  # 如果找不到路径，返回起点和终点

def find_signal_on_path(path_jids, jmap, signal_boxes, max_dist):
    """
    在路径上查找信号名：
    1. 查找与路径线段相交的 signalName
    2. 查找在路径上方的 signalName
    """
    if not path_jids or len(path_jids) < 2:
        return ''

    best_signal = ''
    best_score = float('inf')

    # 构建路径线段
    path_segments = []
    for i in range(len(path_jids) - 1):
        p1 = jmap[path_jids[i]]
        p2 = jmap[path_jids[i + 1]]
        path_segments.append(((p1['x'], p1['y']), (p2['x'], p2['y'])))

    for sig in signal_boxes:
        sx1, sy1, sx2, sy2 = sig['box']
        sig_center = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)

        # 检查1：信号名是否与路径线段相交
        is_intersect = False
        min_dist_to_path = float('inf')

        for seg_start, seg_end in path_segments:
            # 检查信号框是否与线段相交
            if line_intersects_box(seg_start, seg_end, sig['box']):
                is_intersect = True
                break

            # 计算信号中心到线段的距离
            dist_to_seg = point_to_line_distance(sig_center, seg_start, seg_end)
            min_dist_to_path = min(min_dist_to_path, dist_to_seg)

        # 检查2：信号名是否在路径上方
        is_above_path = False
        for seg_start, seg_end in path_segments:
            avg_y = (seg_start[1] + seg_end[1]) / 2
            if sig_center[1] < avg_y - 5:  # 信号在路径上方至少5像素
                is_above_path = True
                break

        # 评分：相交最优，其次是上方且距离近
        if is_intersect:
            score = 0  # 最高优先级
        elif is_above_path and min_dist_to_path < max_dist:
            score = min_dist_to_path
        else:
            continue

        if score < best_score:
            best_score = score
            best_signal = sig['raw_text']

    return best_signal

def line_intersects_box(line_start, line_end, box):
    """检查线段是否与矩形框相交"""
    x1, y1, x2, y2 = box

    # 检查线段的两个端点是否在框内
    def point_in_box(px, py):
        return x1 <= px <= x2 and y1 <= py <= y2

    if point_in_box(line_start[0], line_start[1]) or point_in_box(line_end[0], line_end[1]):
        return True

    # 检查线段是否与矩形的四条边相交
    rect_edges = [
        ((x1, y1), (x2, y1)),  # 上边
        ((x2, y1), (x2, y2)),  # 右边
        ((x2, y2), (x1, y2)),  # 下边
        ((x1, y2), (x1, y1))   # 左边
    ]

    for edge_start, edge_end in rect_edges:
        if segments_intersect(line_start, line_end, edge_start, edge_end):
            return True

    return False

def segments_intersect(p1, p2, p3, p4):
    """检查两条线段是否相交"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    import math

    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)




def find_nearest_signal(px, py, signal_boxes, max_dist):
    """找距离点最近的信号名文字"""
    best, best_d = '', float('inf')
    for sb in signal_boxes:
        d = dist((px, py), sb['center'])
        if d < best_d and d < max_dist:
            best_d = d
            best   = sb['raw_text']
    return best


# ═══════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════

def visualize(image_path, junctions, adj, device_boxes,
              device_entries, connections, out_path, ground_boxes, power_boxes, ground_entries=None, power_entries=None):
    image = cv2.imread(image_path)
    if image is None:
        return

    # 调试：打印图像尺寸
    img_h, img_w = image.shape[:2]
    print(f'\n[DEBUG] 可视化图像尺寸: {img_w} x {img_h}')
    jmap = {j['id']: j for j in junctions}

    # 调试：打印第一个设备框的坐标
    if device_boxes:
        first_dev = device_boxes[0]
        print(f'[DEBUG] 第一个设备框: {first_dev["raw_text"]}')
        print(f'[DEBUG]   Box 坐标: {first_dev["box"]}')
        print(f'[DEBUG]   Center: {first_dev["center"]}')

    # 画所有图的边（灰色细线）
    drawn_edges = set()
    for jid, neighbors in adj.items():
        for nb in neighbors:
            edge = (min(jid,nb), max(jid,nb))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            ja, jb = jmap[jid], jmap[nb]
            cv2.line(image,
                     (int(ja['x']), int(ja['y'])),
                     (int(jb['x']), int(jb['y'])),
                     (210,210,210), 1)

    # 画所有 Junction 节点（灰色小点）
    for j in junctions:
        cv2.circle(image, (int(j['x']), int(j['y'])), 2, (180,180,180), -1)

    # 画设备框（蓝色 BGR: 200,80,0）
    for dev in device_boxes:
        x1,y1,x2,y2 = [int(v) for v in dev['box']]
        cv2.rectangle(image, (x1,y1), (x2,y2), (200,80,0), 2)
        cv2.putText(image, dev['raw_text'], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,60,0), 1)

    # 画Ground框（绿色 BGR: 0,200,0）
    for ground in ground_boxes:
        x1,y1,x2,y2 = [int(v) for v in ground['box']]
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,200,0), 2)
        label = ground.get('raw_text', 'Ground')
        cv2.putText(image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0), 1)

    # 画Power框（红色 BGR: 0,0,200）
    for power in power_boxes:
        x1,y1,x2,y2 = [int(v) for v in power['box']]
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,200), 2)
        label = power.get('raw_text', 'Power')
        cv2.putText(image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,180), 1)

    # 画Device入口节点（青色大圆 BGR: 200,200,0）
    if device_entries:
        all_device_entry_jids = set()
        for jids in device_entries.values():
            all_device_entry_jids.update(jids)
        for jid in all_device_entry_jids:
            if jid in jmap:
                j = jmap[jid]
                cv2.circle(image, (int(j['x']), int(j['y'])), 6, (200,200,0), -1)

    # 画Ground入口节点（黄色大圆 BGR: 0,255,255）
    if ground_entries:
        all_ground_entry_jids = set()
        for jids in ground_entries.values():
            all_ground_entry_jids.update(jids)
        for jid in all_ground_entry_jids:
            if jid in jmap:
                j = jmap[jid]
                cv2.circle(image, (int(j['x']), int(j['y'])), 6, (0,255,255), -1)

    # 画Power入口节点（品红色大圆 BGR: 255,0,255）
    if power_entries:
        all_power_entry_jids = set()
        for jids in power_entries.values():
            all_power_entry_jids.update(jids)
        for jid in all_power_entry_jids:
            if jid in jmap:
                j = jmap[jid]
                cv2.circle(image, (int(j['x']), int(j['y'])), 6, (255,0,255), -1)

    # 画连接关系（红色连线，按实际路径绘制）
    for conn_idx, conn in enumerate(connections):
        path_jids = conn.get('path_jids', [])

        if path_jids and len(path_jids) > 1:
            # 按照实际路径绘制
            for i in range(len(path_jids) - 1):
                p1 = jmap[path_jids[i]]
                p2 = jmap[path_jids[i + 1]]
                cv2.line(image,
                         (int(p1['x']), int(p1['y'])),
                         (int(p2['x']), int(p2['y'])),
                         (0,0,220), 2)

            # 在路径中间位置显示信号名
            mid_idx = len(path_jids) // 2
            mid_point = jmap[path_jids[mid_idx]]
            label = conn['signal'] if conn['signal'] else '?'
            cv2.putText(image, label,
                        (int(mid_point['x']), int(mid_point['y'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,180), 1)
        else:
            # 如果没有路径信息，退化为画直线
            p1 = (int(conn['from_center'][0]), int(conn['from_center'][1]))
            p2 = (int(conn['to_center'][0]), int(conn['to_center'][1]))
            cv2.line(image, p1, p2, (0,0,220), 2)
            mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            label = conn['signal'] if conn['signal'] else '?'
            cv2.putText(image, label, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,180), 1)

    cv2.putText(image,
                f'connections: {len(connections)}',
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,200), 2)
    cv2.imwrite(out_path, image)
    print(f'可视化: {out_path}')

def box_width(box):
    """计算矩形框宽度"""
    return box[2] - box[0]

def box_height(box):
    """计算矩形框高度"""
    return box[3] - box[1]

def boxes_width_similar(box1, box2, width_ratio_threshold=0.3):
    """
    判断两个框的宽度是否相近
    width_ratio_threshold: 宽度差异比例阈值
    """
    w1 = box_width(box1)
    w2 = box_width(box2)

    if w1 == 0 or w2 == 0:
        return False

    # 计算宽度差异比例
    width_diff_ratio = abs(w1 - w2) / max(w1, w2)

    return width_diff_ratio <= width_ratio_threshold


def boxes_horizontally_aligned(box1, box2, overlap_ratio=0.5):
    """
    判断两个框是否在水平方向上对齐（有足够重叠）
    overlap_ratio: 最小重叠比例
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算水平重叠
    overlap_left = max(x1_min, x2_min)
    overlap_right = min(x1_max, x2_max)
    overlap_width = max(0, overlap_right - overlap_left)

    # 取较小的宽度作为基准
    min_width = min(x1_max - x1_min, x2_max - x2_min)

    if min_width == 0:
        return False

    return overlap_width / min_width >= overlap_ratio

def box_area(box):
    """计算矩形框面积"""
    return (box[2] - box[0]) * (box[3] - box[1])


def merge_split_device_boxes(device_boxes, width_ratio_threshold=0.3, vertical_gap_threshold=4000, top_extend=350, bottom_extend=300, max_attempts=3):
    """
    合并分裂的 device 矩形框。

    策略：
    1. 按面积从大到小尝试多个参考框（默认最多尝试2个不重叠的框）
    2. 对每个参考框，先在原始范围内寻找可合并的框
    3. 如果没找到，扩展参考框的上下边界后再寻找
    4. 一旦找到可合并的框组，立即执行合并并返回
    5. 如果某个参考框最终没有合并任何框，但进行了扩展搜索，则对该框本身进行边界扩展
    6. 固定尝试 max_attempts 个不重叠的框后才退出
    7. 如果候选框与已处理的框重叠，则跳过并继续查找下一个

    参数：
    - width_ratio_threshold: 宽度相似度阈值（默认0.3，即宽度差异不超过30%）
    - vertical_gap_threshold: 垂直间隙阈值（默认200像素）
    - top_extend: 扩展后在顶部额外增加的距离（默认375像素）
    - bottom_extend: 扩展后在底部额外增加的距离（默认85像素）
    - max_attempts: 固定尝试的不重叠参考框数量（默认2个）

    返回：
    - 合并后的 device_boxes
    """
    if len(device_boxes) <= 1:
        return device_boxes

    print(f'\n[Box Merge] 合并前: {len(device_boxes)} 个 device 框')

    # 按面积降序排序
    indexed_boxes = [(idx, dev, box_area(dev['box'])) for idx, dev in enumerate(device_boxes)]
    indexed_boxes.sort(key=lambda x: x[2], reverse=True)

    # 记录已经处理过的框索引（包括合并的和扩展的）
    processed_indices = set()

    # 计数器：成功尝试的不重叠框数量
    successful_attempts = 0

    # 遍历所有候选框，直到找到 max_attempts 个不重叠的框
    for ref_idx, ref_dev, ref_area in indexed_boxes:
        # 如果已经达到目标尝试次数，退出
        if successful_attempts >= max_attempts:
            break

        # 跳过已经处理过的框
        if ref_idx in processed_indices:
            continue

        print(f'\n[Box Merge] === 尝试第 {successful_attempts + 1}/{max_attempts} 个参考框 (idx={ref_idx}) ===')

        ref_box = ref_dev['box']
        ref_x1, ref_y1, ref_x2, ref_y2 = ref_box
        ref_width = box_width(ref_box)
        ref_center_x = (ref_x1 + ref_x2) / 2

        print(f'[Box Merge] 参考框: {ref_box}, 面积={ref_area:.0f}, 宽度={ref_width:.0f}')

        # 第一轮：在原始范围内寻找
        group_indices = _find_mergeable_boxes(
            device_boxes, ref_idx, ref_box, ref_width, ref_center_x,
            width_ratio_threshold, vertical_gap_threshold
        )

        expanded_search = False

        # 如果没找到，扩展范围后再寻找
        if len(group_indices) <= 1:
            print(f'[Box Merge] 未找到可合并框，扩展搜索范围...')
            expanded_search = True

            # 创建扩展后的参考框
            extended_ref_box = [
                ref_x1,
                ref_y1 - vertical_gap_threshold,  # 向上扩展
                ref_x2,
                ref_y2 + vertical_gap_threshold   # 向下扩展
            ]

            print(f'[Box Merge] 扩展后的参考框: {extended_ref_box}')

            # 第二轮：在扩展范围内寻找
            group_indices = _find_mergeable_boxes(
                device_boxes, ref_idx, extended_ref_box, ref_width, ref_center_x,
                width_ratio_threshold, vertical_gap_threshold * 2  # 允许更大的间隙
            )

        # 标记该框已处理
        processed_indices.add(ref_idx)
        successful_attempts += 1

        # 如果找到了可合并的框
        if len(group_indices) > 1:
            print(f'[Box Merge] ✓ 找到 {len(group_indices)} 个可合并的框')

            # 创建合并后的新框
            merged_x1 = min(device_boxes[idx]['box'][0] for idx in group_indices)
            merged_y1 = min(device_boxes[idx]['box'][1] for idx in group_indices)
            merged_x2 = max(device_boxes[idx]['box'][2] for idx in group_indices)
            merged_y2 = max(device_boxes[idx]['box'][3] for idx in group_indices)

            # 在顶部和底部额外扩展
            merged_y1 = merged_y1 - top_extend
            merged_y2 = merged_y2 + bottom_extend

            merged_box = [merged_x1, merged_y1, merged_x2, merged_y2]

            # 合并 raw_text（取非空的文本）
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

            # 构建新的 device_boxes：合并后的框 + 未参与合并的其他框
            new_device_boxes = [merged_entry]
            for idx, dev in enumerate(device_boxes):
                if idx not in group_indices:
                    new_device_boxes.append(dev)

            print(f'[Box Merge] 合并结果: {merged_box} (顶部扩展{top_extend}px, 底部扩展{bottom_extend}px)')
            print(f'[Box Merge] 合并后: {len(new_device_boxes)} 个 device 框\n')

            return new_device_boxes
        else:
            print(f'[Box Merge] ✗ 第 {successful_attempts} 个参考框未找到可合并的框')

            # 如果进行了扩展搜索但仍没找到，对这个框本身进行边界扩展
            if expanded_search:
                print(f'[Box Merge] 对该框进行边界扩展: 顶部+{top_extend}px, 底部+{bottom_extend}px')

                # 直接修改原设备框的边界
                device_boxes[ref_idx]['box'][0] = ref_x1
                device_boxes[ref_idx]['box'][1] = ref_y1 - top_extend  # 向上扩展
                device_boxes[ref_idx]['box'][2] = ref_x2
                device_boxes[ref_idx]['box'][3] = ref_y2 + bottom_extend  # 向下扩展

                # 更新中心点
                device_boxes[ref_idx]['center'] = box_center(device_boxes[ref_idx]['box'])

                print(f'[Box Merge] 扩展后的框: {device_boxes[ref_idx]["box"]}')

    # 固定尝试完 max_attempts 个不重叠的框后退出
    print(f'\n[Box Merge] 已完成 {successful_attempts} 次有效尝试（目标 {max_attempts} 次）')
    print(f'[Box Merge] 最终结果: {len(device_boxes)} 个 device 框（部分可能已扩展）\n')
    return device_boxes

def _find_mergeable_boxes(device_boxes, ref_idx, ref_box, ref_width, ref_center_x,
                          width_ratio_threshold, vertical_gap_threshold):
    """
    寻找可与参考框合并的框

    返回：
    - 可合并框的索引列表（包含参考框本身）
    """
    ref_x1, ref_y1, ref_x2, ref_y2 = ref_box
    group_indices = [ref_idx]

    for idx, dev in enumerate(device_boxes):
        if idx == ref_idx:
            continue

        cand_box = dev['box']
        cand_x1, cand_y1, cand_x2, cand_y2 = cand_box
        cand_width = box_width(cand_box)
        cand_center_x = (cand_x1 + cand_x2) / 2

        # 检查宽度是否相近
        if not boxes_width_similar(ref_box, cand_box, width_ratio_threshold):
            continue

        # 检查水平位置是否对齐（中心点x坐标应该接近）
        x_center_diff = abs(cand_center_x - ref_center_x)
        x_alignment_threshold = ref_width * 0.15  # 中心点偏移不超过宽度的15%

        if x_center_diff > x_alignment_threshold:
            continue

        # 检查垂直位置关系（上方或下方）
        is_above = cand_y2 < ref_y1  # 候选框在参考框上方
        is_below = cand_y1 > ref_y2  # 候选框在参考框下方

        if is_above or is_below:
            # 计算垂直距离
            if is_above:
                vertical_dist = ref_y1 - cand_y2
                position = "上方"
            else:
                vertical_dist = cand_y1 - ref_y2
                position = "下方"

            # 只要垂直距离在阈值内就合并
            if vertical_dist <= vertical_gap_threshold:
                group_indices.append(idx)
                print(f'[Box Merge]   找到{position}可合并框 (idx={idx}): {cand_box}, 宽度={cand_width:.0f}, 垂直距离={vertical_dist:.0f}')

    return group_indices

# ═══════════════════════════════════════════
#  保存结果
# ═══════════════════════════════════════════

def save_results(connections, out_dir, base):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, base+'_connections.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['信号名', '设备A', '设备B'])
        for c in connections:
            w.writerow([c['signal'], c['from_device'], c['to_device']])
    print(f'CSV: {csv_path}')

    json_path = os.path.join(out_dir, base+'_connections.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([{
            'signal':      c['signal'],
            'from_device': c['from_device'],
            'to_device':   c['to_device'],
        } for c in connections], f, indent=2, ensure_ascii=False)
    print(f'JSON: {json_path}')


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════

def process(image_path, out_dir='./results'):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    print('\n=== Step 1: HAWP 线段检测 ===')
    model = load_hawp()
    hawp_lines, ori_w, ori_h = hawp_predict(model, image_path)

    print('\n=== Step 2: 构建图 ===')
    junctions, adj = build_graph(hawp_lines, MERGE_TH)

    print('\n=== Step 3: YOLO + OCR ===')
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    yolo_ocr_objects = process_single_page_image(img_b64)
    device_boxes, ground_boxes, power_boxes, signal_boxes = parse_objects(yolo_ocr_objects)

    print('\n=== Step 4: 找设备入口节点 ===')
    # 每个设备对应的入口点
    device_entries, ground_entries, power_entries, entry_to_lines = find_entry_points(junctions, device_boxes, EDGE_TOL, ground_boxes, power_boxes, hawp_lines)
    print('\n=== Step 4.5: 重建图（包含入口点）===')
    adj = rebuild_graph_with_entries(junctions, adj, [], entry_to_lines, hawp_lines, MERGE_TH)

    print('\n=== Step 5: BFS 搜索连接关系 ===')
    connections = find_connections_bfs(
        junctions, adj, device_boxes,
        device_entries, signal_boxes, OCR_DIST,ground_entries, ground_boxes, power_entries, power_boxes)

    print('\n--- 连接关系预览 ---')
    for c in connections:
        sig = f'[{c["signal"]}]' if c['signal'] else '[未知信号]'
        print(f'  {c["from_device"]}  ←{sig}→  {c["to_device"]}')

    print('\n=== Step 6: 保存结果 ===')
    visualize(image_path, junctions, adj, device_boxes,
              device_entries, connections,
              os.path.join(out_dir, base+'_result.jpg'),ground_boxes,power_boxes,ground_entries,power_entries)
    save_results(connections, out_dir, base)

    return connections




# ═══════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════

if __name__ == '__main__':
    # 配置参数
    IMAGE_PATH = r'F:\office\pythonProjects\SystemVision-原理图识别\yolo\images\page_28_original.jpg'  # 修改为您的图片路径
    OUTPUT_DIR = './output'                       # 输出目录

    process(IMAGE_PATH, OUTPUT_DIR)