"""
图构建服务：端点合并、邻接表构建、入口点检测
"""
import math
import numpy as np
from collections import defaultdict
from app.core.config import Config


def dist(p1, p2):
    """计算两点距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def box_center(box):
    """计算矩形框中心"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def line_segment_intersection(p1, p2, p3, p4):
    """计算两条线段的交点"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None


def line_box_intersections(line_start, line_end, box):
    """计算线段与矩形框的所有交点"""
    x1, y1, x2, y2 = box
    intersections = []

    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1))
    ]

    for edge_start, edge_end in edges:
        point = line_segment_intersection(line_start, line_end, edge_start, edge_end)
        if point is not None:
            intersections.append(point)

    return intersections


class GraphBuilder:
    """图构建器"""

    @staticmethod
    def build_graph(hawp_lines, merge_threshold):
        """
        端点合并（Union-Find）+ 构建邻接表

        Returns:
            tuple: (junctions, adj)
        """
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

        for i in range(n):
            for j in range(i + 1, n):
                if dist(raw_pts[i], raw_pts[j]) < merge_threshold:
                    union(i, j)

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

        adj = defaultdict(set)
        for idx, seg in enumerate(hawp_lines):
            pi, pj = idx * 2, idx * 2 + 1
            ja = root_to_jid[find(pi)]
            jb = root_to_jid[find(pj)]
            if ja != jb:
                adj[ja].add(jb)
                adj[jb].add(ja)

        print(f'构建图：{len(junctions)} 个节点，'
              f'{sum(len(v) for v in adj.values()) // 2} 条边')

        return junctions, dict(adj)

    @staticmethod
    def find_entry_points(junctions, device_boxes, ground_boxes, power_boxes,
                          hawp_lines, box_expand=None):
        """
        查找设备入口点

        Returns:
            tuple: (device_entries, ground_entries, power_entries, entry_to_lines)
        """
        if box_expand is None:
            box_expand = Config.BOX_EXPAND

        device_entries = defaultdict(list)
        ground_entries = defaultdict(list)
        power_entries = defaultdict(list)

        new_junctions = []
        next_jid = len(junctions)
        entry_to_lines = []

        def collect_intersections(boxes, entries_dict):
            nonlocal next_jid

            for box_idx, box_info in enumerate(boxes):
                original_box = box_info['box']

                expanded_box = [
                    max(0, original_box[0] - box_expand),
                    max(0, original_box[1] - box_expand),
                    original_box[2] + box_expand,
                    original_box[3] + box_expand
                ]

                entry_jids = []

                for line_idx, line in enumerate(hawp_lines):
                    p1 = line['p1']
                    p2 = line['p2']

                    intersections = line_box_intersections(p1, p2, expanded_box)

                    for ix, iy in intersections:
                        jid = next_jid
                        next_jid += 1
                        new_junction = {'id': jid, 'x': ix, 'y': iy}
                        new_junctions.append(new_junction)
                        entry_jids.append(jid)
                        entry_to_lines.append((jid, line_idx))

                if entry_jids:
                    entries_dict[box_idx] = entry_jids

        collect_intersections(device_boxes, device_entries)
        collect_intersections(ground_boxes, ground_entries)
        collect_intersections(power_boxes, power_entries)

        junctions.extend(new_junctions)

        print(f'\n框扩展距离: {box_expand} 像素')
        print(f'新增入口点数量: {len(new_junctions)}')
        print(f'总junction数量: {len(junctions)}')

        return dict(device_entries), dict(ground_entries), dict(power_entries), entry_to_lines

    @staticmethod
    def rebuild_graph_with_entries(original_junctions, original_adj, entry_to_lines,
                                   hawp_lines, merge_threshold):
        """
        重新构建图，将入口点插入到HAWP线段中
        """
        new_adj = defaultdict(set)
        for jid, neighbors in original_adj.items():
            for nb in neighbors:
                new_adj[jid].add(nb)
                new_adj[nb].add(jid)

        line_to_entries = defaultdict(list)
        for entry_jid, line_idx in entry_to_lines:
            line_to_entries[line_idx].append(entry_jid)

        jmap = {j['id']: j for j in original_junctions}

        cut_count = 0
        skip_count = 0

        for line_idx, entry_jids in line_to_entries.items():
            if line_idx >= len(hawp_lines):
                continue

            line = hawp_lines[line_idx]
            p1 = np.array(line['p1'])
            p2 = np.array(line['p2'])

            ep1_jid = None
            ep2_jid = None
            min_dist1 = float('inf')
            min_dist2 = float('inf')

            for jid, j in jmap.items():
                jpos = np.array([j['x'], j['y']])
                d1 = np.linalg.norm(jpos - p1)
                d2 = np.linalg.norm(jpos - p2)

                if d1 < min_dist1:
                    min_dist1 = d1
                    ep1_jid = jid

                if d2 < min_dist2 and d2 < merge_threshold * 2:
                    min_dist2 = d2
                    ep2_jid = jid

            if ep1_jid is None or ep2_jid is None:
                skip_count += 1
                continue

            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)

            if line_len < 1e-6:
                continue

            line_unit = line_vec / line_len

            entry_positions = []
            for entry_jid in entry_jids:
                entry_pos = np.array([jmap[entry_jid]['x'], jmap[entry_jid]['y']])
                proj = np.dot(entry_pos - p1, line_unit) / line_len
                proj = np.clip(proj, 0.0, 1.0)
                entry_positions.append((proj, entry_jid))

            entry_positions.sort(key=lambda x: x[0])

            if ep2_jid in new_adj.get(ep1_jid, set()):
                new_adj[ep1_jid].discard(ep2_jid)
                new_adj[ep2_jid].discard(ep1_jid)

            prev_jid = ep1_jid
            for proj, entry_jid in entry_positions:
                new_adj[prev_jid].add(entry_jid)
                new_adj[entry_jid].add(prev_jid)
                prev_jid = entry_jid
                cut_count += 1

            new_adj[prev_jid].add(ep2_jid)
            new_adj[ep2_jid].add(prev_jid)
            cut_count += 1

        print(f'[重建图] 成功切割 {cut_count} 条线段连接')
        print(f'[重建图] 跳过 {skip_count} 条无法匹配的线段')
        print(f'[重建图] 总边数: {sum(len(v) for v in new_adj.values()) // 2}')

        return dict(new_adj)
