"""
连接关系查找服务：路径搜索、信号名匹配
"""
import math
from collections import deque, defaultdict
from app.core.config import Config


def boxes_edge_touch(box1, box2, touch_threshold=None):
    """判断两个矩形框的边是否相交或触碰"""
    if touch_threshold is None:
        touch_threshold = Config.TOUCH_THRESHOLD

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    is_box2_inside = (x2_min >= x1_min and x2_max <= x1_max and
                      y2_min >= y1_min and y2_max <= y1_max)
    is_box1_inside = (x1_min >= x2_min and x1_max <= x2_max and
                      y1_min >= y2_min and y1_max <= y2_max)

    if is_box2_inside or is_box1_inside:
        return False

    expanded_x2_min = x2_min - touch_threshold
    expanded_y2_min = y2_min - touch_threshold
    expanded_x2_max = x2_max + touch_threshold
    expanded_y2_max = y2_max + touch_threshold

    has_overlap_x = not (x1_max < expanded_x2_min or x1_min > expanded_x2_max)
    has_overlap_y = not (y1_max < expanded_y2_min or y1_min > expanded_y2_max)

    return has_overlap_x and has_overlap_y


def segments_intersect(p1, p2, p3, p4):
    """检查两条线段是否相交"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def line_intersects_box(line_start, line_end, box):
    """检查线段是否与矩形框相交"""
    x1, y1, x2, y2 = box

    def point_in_box(px, py):
        return x1 <= px <= x2 and y1 <= py <= y2

    if point_in_box(line_start[0], line_start[1]) or point_in_box(line_end[0], line_end[1]):
        return True

    rect_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1))
    ]

    for edge_start, edge_end in rect_edges:
        if segments_intersect(line_start, line_end, edge_start, edge_end):
            return True

    return False


def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
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


class ConnectionFinder:
    """连接关系查找器"""

    @staticmethod
    def find_all_paths_dfs(start_jids, target_set, adj, exclude_jids, max_depth=100):
        """使用DFS找所有从起点集合到目标集合的路径"""
        all_paths = []
        paths_found = set()

        def dfs(current, target, path, visited):
            if len(path) > max_depth:
                return

            if current in target:
                path_tuple = tuple(path)
                if path_tuple not in paths_found:
                    paths_found.add(path_tuple)
                    all_paths.append(list(path))
                return

            for neighbor in adj.get(current, []):
                if neighbor in visited:
                    continue
                if neighbor in exclude_jids:
                    continue

                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, path, visited)
                path.pop()
                visited.remove(neighbor)

        for start_jid in start_jids:
            visited = {start_jid}
            dfs(start_jid, target_set, [start_jid], visited)

        print(f'[DFS] 找到 {len(all_paths)} 条路径')
        return all_paths

    @staticmethod
    def find_signal_on_path(path_jids, jmap, signal_boxes, max_dist):
        """在路径上查找信号名"""
        if not path_jids or len(path_jids) < 2:
            return ''

        best_signal = ''
        best_score = float('inf')

        path_segments = []
        for i in range(len(path_jids) - 1):
            p1 = jmap[path_jids[i]]
            p2 = jmap[path_jids[i + 1]]
            path_segments.append(((p1['x'], p1['y']), (p2['x'], p2['y'])))

        for sig in signal_boxes:
            sx1, sy1, sx2, sy2 = sig['box']
            sig_center = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)

            is_intersect = False
            min_dist_to_path = float('inf')

            for seg_start, seg_end in path_segments:
                if line_intersects_box(seg_start, seg_end, sig['box']):
                    is_intersect = True
                    break

                dist_to_seg = point_to_line_distance(sig_center, seg_start, seg_end)
                min_dist_to_path = min(min_dist_to_path, dist_to_seg)

            is_above_path = False
            for seg_start, seg_end in path_segments:
                avg_y = (seg_start[1] + seg_end[1]) / 2
                if sig_center[1] < avg_y - 5:
                    is_above_path = True
                    break

            if is_intersect:
                score = 0
            elif is_above_path and min_dist_to_path < max_dist:
                score = min_dist_to_path
            else:
                continue

            if score < best_score:
                best_score = score
                best_signal = sig['raw_text']

        return best_signal

    @staticmethod
    def find_connections(junctions, adj, device_boxes, device_entries, signal_boxes,
                         ocr_dist, ground_entries, ground_boxes, power_entries, power_boxes):
        """
        查找所有设备间的连接关系

        Returns:
            list: 连接关系列表
        """
        jmap = {j['id']: j for j in junctions}

        all_devices = []
        all_entries = {}
        offset = 0

        for idx, dev in enumerate(device_boxes):
            dev['type'] = 'device'
            all_devices.append(dev)
            if idx in device_entries:
                all_entries[idx + offset] = device_entries[idx]
        device_count = len(device_boxes)
        offset += device_count

        for idx, ground in enumerate(ground_boxes):
            ground['type'] = 'ground'
            all_devices.append(ground)
            if idx in ground_entries:
                all_entries[idx + offset] = ground_entries[idx]
        ground_count = len(ground_boxes)
        offset += ground_count

        for idx, power in enumerate(power_boxes):
            power['type'] = 'power'
            all_devices.append(power)
            if idx in power_entries:
                all_entries[idx + offset] = power_entries[idx]

        n_devs = len(all_devices)
        connections = []

        print(f'\n[BFS] 开始搜索连接关系，共 {n_devs} 个设备（{device_count} device, '
              f'{ground_count} ground, {len(power_boxes)} power）')

        skipped_edge_touch = 0

        all_entry_jids = set()
        for entries in all_entries.values():
            all_entry_jids.update(entries)

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

                if boxes_edge_touch(dev_a_info['box'], dev_b_info['box']):
                    skipped_edge_touch += 1
                    print(f'[跳过] 设备框边相交: "{dev_a_info["raw_text"]}" 和 "{dev_b_info["raw_text"]}"')
                    continue

                target_set = set(entries_b)

                all_paths = ConnectionFinder.find_all_paths_dfs(
                    entries_a,
                    target_set,
                    adj,
                    exclude_jids=all_entry_jids - target_set,
                    max_depth=100
                )

                for path_jids in all_paths:
                    if len(path_jids) < 2:
                        continue

                    entry_jid = path_jids[0]
                    reached_jid = path_jids[-1]

                    signal = ConnectionFinder.find_signal_on_path(
                        path_jids, jmap, signal_boxes, ocr_dist
                    )

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

        print(f'[BFS] 跳过 {skipped_edge_touch} 对边相交的设备框')
        print(f'[BFS] 找到 {len(connections)} 条连接路径')

        return connections
