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

def match_device_names(device_boxes, signal_boxes, top_margin=50, bottom_margin=30):
    """
    为每个 device 匹配一个最合适的 signalName 作为 raw_text。
    """
    for dev in device_boxes:
        dx1, dy1, dx2, dy2 = dev['box']

        # 定义搜索区域的垂直范围 (以顶部边 dy1 为基准)
        search_y_min = dy1 - top_margin
        search_y_max = dy1 + bottom_margin

        best_signal = None
        min_dist = float('inf')

        for sig in signal_boxes:
            sx1, sy1, sx2, sy2 = sig['box']
            sig_center_x = (sx1 + sx2) / 2
            sig_center_y = (sy1 + sy2) / 2

            # 1. 垂直距离检查
            if not (search_y_min <= sig_center_y <= search_y_max):
                continue

            # 2. 水平对齐检查 (信号中心必须在设备宽度的左右边界内)
            if not (dx1 <= sig_center_x <= dx2):
                continue

            # 3. 计算到顶部边的垂直距离
            dist_to_top_edge = abs(sig_center_y - dy1)

            # 4. 优先级判断：优先选框内的，其次选最近的
            is_inside = (dy1 <= sy1 and sy2 <= dy2)

            if best_signal is None:
                best_signal = sig
                min_dist = dist_to_top_edge
            else:
                was_inside = (best_signal['box'][1] >= dy1 and best_signal['box'][3] <= dy2)

                if is_inside and not was_inside:
                    best_signal = sig
                    min_dist = dist_to_top_edge
                elif is_inside == was_inside:
                    if dist_to_top_edge < min_dist:
                        best_signal = sig
                        min_dist = dist_to_top_edge

        # 直接更新 raw_text
        if best_signal:
            dev['raw_text'] = best_signal['raw_text']
        else:
            dev['raw_text'] = "Unknown"

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

def find_entry_points(junctions, device_boxes, edge_tol, ground_boxes, power_boxes):
    """
    找每个设备框、ground框、power框边缘附近的 Junction（入口节点）。
    返回 {dev_idx: [jid, ...]}, {ground_idx: [jid, ...]}, {power_idx: [jid, ...]}
    """
    device_entries = defaultdict(list)
    ground_entries = defaultdict(list)
    power_entries = defaultdict(list)

    for j in junctions:
        # 检查 device 入口点
        for dev_idx, dev in enumerate(device_boxes):
            if point_near_box_edge(j['x'], j['y'], dev['box'], edge_tol):
                device_entries[dev_idx].append(j['id'])

        # 检查 ground 入口点
        for ground_idx, ground in enumerate(ground_boxes):
            if point_near_box_edge(j['x'], j['y'], ground['box'], edge_tol):
                ground_entries[ground_idx].append(j['id'])

        # 检查 power 入口点
        for power_idx, power in enumerate(power_boxes):
            if point_near_box_edge(j['x'], j['y'], power['box'], edge_tol):
                power_entries[power_idx].append(j['id'])

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

    return dict(device_entries), dict(ground_entries), dict(power_entries)


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


def find_connections_bfs(junctions, adj, device_boxes,
                         device_entries, signal_boxes,
                         ocr_dist,ground_entries, ground_boxes, power_entries,power_boxes):
    """
    对每对设备，从设备A的入口节点BFS搜索，
    看能否到达设备B的入口节点。
    """
    # id对应的x,y坐标字典
    jmap = {j['id']: j for j in junctions}
    n_devs = len(device_boxes)
    connections = []
    # 已经检测过的设备对，避免重复检测同一设备之间的连接
    seen_pairs  = set()

    # 预计算：哪些 junction 在设备框内部（BFS时避免穿越）
    interior_jids = set()
    for j in junctions:
        for dev in device_boxes:
            if point_in_box(j['x'], j['y'], dev['box'], pad=-5):
                interior_jids.add(j['id'])

    for dev_a in range(n_devs):
        # 安全的获取设备对应的junction
        entries_a = device_entries.get(dev_a, [])
        if not entries_a:
            continue

        for dev_b in range(dev_a+1, n_devs):
            entries_b = device_entries.get(dev_b, [])
            if not entries_b:
                continue

            # 创建元组作为已检测设备的键
            pair = (dev_a, dev_b)
            if pair in seen_pairs:
                continue

            target_set = set(entries_b)
            found = False

            # TODO 进入BFS前，可根据两个框的相对位置方向排除某些junction，比如B框在A框下面，
            #  则只需要A框的左右下边的junction和B框的左右上边的junction进行搜索
            for entry_jid in entries_a:
                # 返回的是从起始点到终点的的某个具体的点，也就是用电器B位置中的某个点
                ok, reached_jid = bfs_path_exists(
                    entry_jid,
                    target_set,
                    adj,
                    exclude_jids=interior_jids - target_set,
                    max_depth=30
                )
                if ok:
                    # 找到连接，查找附近的信号名
                    j_start = jmap[entry_jid]
                    signal  = find_nearest_signal(
                        j_start['x'], j_start['y'],
                        signal_boxes, ocr_dist)

                    connections.append({
                        'from_device':  device_boxes[dev_a]['raw_text'],
                        'to_device':    device_boxes[dev_b]['raw_text'],
                        'signal':       signal,
                        'from_entry':   entry_jid,
                        'to_entry':     reached_jid,
                        'from_center':  (j_start['x'], j_start['y']),
                        'to_center':    (jmap[reached_jid]['x'],
                                         jmap[reached_jid]['y']),
                    })
                    seen_pairs.add(pair)
                    found = True
                    break

    print(f'找到 {len(connections)} 对连接关系')
    return connections


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
              device_entries, connections, out_path):
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

    # 画设备框（蓝色）
    for dev in device_boxes:
        x1,y1,x2,y2 = [int(v) for v in dev['box']]
        cv2.rectangle(image, (x1,y1), (x2,y2), (200,80,0), 2)
        cv2.putText(image, dev['raw_text'], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,60,0), 1)

    # 画入口节点（绿色大圆）
    all_entry_jids = set()
    for jids in device_entries.values():
        all_entry_jids.update(jids)
    for jid in all_entry_jids:
        j = jmap[jid]
        cv2.circle(image, (int(j['x']), int(j['y'])), 6, (0,200,80), -1)

    # 画连接关系（红色连线）
    for conn in connections:
        p1 = (int(conn['from_center'][0]), int(conn['from_center'][1]))
        p2 = (int(conn['to_center'][0]),   int(conn['to_center'][1]))
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
    device_entries, ground_entries, power_entries = find_entry_points(junctions, device_boxes, EDGE_TOL, ground_boxes, power_boxes)
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
              os.path.join(out_dir, base+'_result.jpg'))
    save_results(connections, out_dir, base)

    return connections

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


def merge_split_device_boxes(device_boxes, width_ratio_threshold=0.3, vertical_gap_threshold=1000):
    """
    合并分裂的 device 矩形框。

    策略：
    1. 找到面积最大的框作为参考框
    2. 在参考框上方和下方寻找与之宽度相近、水平对齐的框
    3. 只将这些符合条件的框合并为一个大框，其他框保持不变

    参数：
    - width_ratio_threshold: 宽度相似度阈值（默认0.3，即宽度差异不超过30%）
    - vertical_gap_threshold: 垂直间隙阈值（默认50像素）

    返回：
    - 合并后的 device_boxes
    """
    if len(device_boxes) <= 1:
        return device_boxes

    print(f'\n[Box Merge] 合并前: {len(device_boxes)} 个 device 框')

    # 找到面积最大的框
    max_area_idx = 0
    max_area = 0
    for idx, dev in enumerate(device_boxes):
        area = box_area(dev['box'])
        if area > max_area:
            max_area = area
            max_area_idx = idx

    ref_box = device_boxes[max_area_idx]['box']
    ref_x1, ref_y1, ref_x2, ref_y2 = ref_box
    ref_width = box_width(ref_box)

    print(f'[Box Merge] 面积最大的框 (idx={max_area_idx}): {ref_box}, 面积={max_area:.0f}, 宽度={ref_width:.0f}')

    # 寻找可以合并的框（只在最大框的上方和下方）
    group_indices = [max_area_idx]

    for idx, dev in enumerate(device_boxes):
        if idx == max_area_idx:
            continue

        cand_box = dev['box']
        cand_x1, cand_y1, cand_x2, cand_y2 = cand_box
        cand_width = box_width(cand_box)

        # 检查宽度是否相近
        if not boxes_width_similar(ref_box, cand_box, width_ratio_threshold):
            continue

        # 检查水平位置是否对齐（x坐标应该接近）
        x_diff_left = abs(cand_x1 - ref_x1)
        x_diff_right = abs(cand_x2 - ref_x2)
        x_alignment_threshold = ref_width * 0.2  # x偏移不超过宽度的20%

        if x_diff_left > x_alignment_threshold or x_diff_right > x_alignment_threshold:
            continue

        # 检查垂直位置关系（上方或下方）
        vertical_gap_above = ref_y1 - cand_y2  # 候选框在参考框上方
        vertical_gap_below = cand_y1 - ref_y2  # 候选框在参考框下方

        # 判断是否是上下相邻（允许小的重叠或间隙）
        is_above = -10 <= vertical_gap_above <= vertical_gap_threshold
        is_below = -10 <= vertical_gap_below <= vertical_gap_threshold

        if is_above or is_below:
            group_indices.append(idx)
            position = "上方" if is_above else "下方"
            print(f'[Box Merge]   找到{position}可合并框 (idx={idx}): {cand_box}, 宽度={cand_width:.0f}')

    # 如果找到了可合并的框
    if len(group_indices) > 1:
        # 创建合并后的新框
        merged_x1 = min(device_boxes[idx]['box'][0] for idx in group_indices)
        merged_y1 = min(device_boxes[idx]['box'][1] for idx in group_indices)
        merged_x2 = max(device_boxes[idx]['box'][2] for idx in group_indices)
        merged_y2 = max(device_boxes[idx]['box'][3] for idx in group_indices)
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

        print(f'[Box Merge] 合并 {len(group_indices)} 个框 -> {merged_box}')
        print(f'[Box Merge] 合并后: {len(new_device_boxes)} 个 device 框\n')

        return new_device_boxes
    else:
        print(f'[Box Merge] 未找到可合并的框，保持原样')
        print(f'[Box Merge] 合并后: {len(device_boxes)} 个 device 框\n')
        return device_boxes


# ═══════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════

if __name__ == '__main__':
    # 配置参数
    IMAGE_PATH = r'F:\office\pythonProjects\SystemVision-原理图识别\yolo\images\page_3_original.jpg'  # 修改为您的图片路径
    OUTPUT_DIR = './output'                       # 输出目录

    process(IMAGE_PATH, OUTPUT_DIR)