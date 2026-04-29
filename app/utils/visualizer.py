"""
可视化工具
"""
import cv2
from app.core.config import Config


class Visualizer:
    """可视化工具类"""

    @staticmethod
    def visualize(image_path, junctions, adj, device_boxes, device_entries,
                  connections, out_path, ground_boxes, power_boxes,
                  ground_entries=None, power_entries=None):
        """生成可视化结果图"""
        image = cv2.imread(image_path)
        if image is None:
            return

        img_h, img_w = image.shape[:2]
        print(f'\n[DEBUG] 可视化图像尺寸: {img_w} x {img_h}')

        jmap = {j['id']: j for j in junctions}

        if device_boxes:
            first_dev = device_boxes[0]
            print(f'[DEBUG] 第一个设备框: {first_dev["raw_text"]}')
            print(f'[DEBUG]   Box 坐标: {first_dev["box"]}')
            print(f'[DEBUG]   Center: {first_dev["center"]}')

        # 画所有图的边（灰色细线）
        drawn_edges = set()
        for jid, neighbors in adj.items():
            for nb in neighbors:
                edge = (min(jid, nb), max(jid, nb))
                if edge in drawn_edges:
                    continue
                drawn_edges.add(edge)
                ja, jb = jmap[jid], jmap[nb]
                cv2.line(image,
                         (int(ja['x']), int(ja['y'])),
                         (int(jb['x']), int(jb['y'])),
                         (210, 210, 210), 1)

        # 画所有 Junction 节点（灰色小点）
        for j in junctions:
            cv2.circle(image, (int(j['x']), int(j['y'])), 2, (180, 180, 180), -1)

        # 画设备框（蓝色）
        for dev in device_boxes:
            x1, y1, x2, y2 = [int(v) for v in dev['box']]
            cv2.rectangle(image, (x1, y1), (x2, y2), (200, 80, 0), 2)
            cv2.putText(image, dev['raw_text'], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 60, 0), 1)

        # 画Ground框（绿色）
        for ground in ground_boxes:
            x1, y1, x2, y2 = [int(v) for v in ground['box']]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = ground.get('raw_text', 'Ground')
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

        # 画Power框（红色）
        for power in power_boxes:
            x1, y1, x2, y2 = [int(v) for v in power['box']]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 2)
            label = power.get('raw_text', 'Power')
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)

        # 画Device入口节点（青色大圆）
        if device_entries:
            all_device_entry_jids = set()
            for jids in device_entries.values():
                all_device_entry_jids.update(jids)
            for jid in all_device_entry_jids:
                if jid in jmap:
                    j = jmap[jid]
                    cv2.circle(image, (int(j['x']), int(j['y'])), 6, (200, 200, 0), -1)

        # 画Ground入口节点（黄色大圆）
        if ground_entries:
            all_ground_entry_jids = set()
            for jids in ground_entries.values():
                all_ground_entry_jids.update(jids)
            for jid in all_ground_entry_jids:
                if jid in jmap:
                    j = jmap[jid]
                    cv2.circle(image, (int(j['x']), int(j['y'])), 6, (0, 255, 255), -1)

        # 画Power入口节点（品红色大圆）
        if power_entries:
            all_power_entry_jids = set()
            for jids in power_entries.values():
                all_power_entry_jids.update(jids)
            for jid in all_power_entry_jids:
                if jid in jmap:
                    j = jmap[jid]
                    cv2.circle(image, (int(j['x']), int(j['y'])), 6, (255, 0, 255), -1)

        # 画连接关系（红色连线）
        for conn_idx, conn in enumerate(connections):
            path_jids = conn.get('path_jids', [])

            if path_jids and len(path_jids) > 1:
                for i in range(len(path_jids) - 1):
                    p1 = jmap[path_jids[i]]
                    p2 = jmap[path_jids[i + 1]]
                    cv2.line(image,
                             (int(p1['x']), int(p1['y'])),
                             (int(p2['x']), int(p2['y'])),
                             (0, 0, 220), 2)

                mid_idx = len(path_jids) // 2
                mid_point = jmap[path_jids[mid_idx]]
                label = conn['signal'] if conn['signal'] else '?'
                cv2.putText(image, label,
                            (int(mid_point['x']), int(mid_point['y'])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
            else:
                p1 = (int(conn['from_center'][0]), int(conn['from_center'][1]))
                p2 = (int(conn['to_center'][0]), int(conn['to_center'][1]))
                cv2.line(image, p1, p2, (0, 0, 220), 2)
                mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                label = conn['signal'] if conn['signal'] else '?'
                cv2.putText(image, label, (mx, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)

        cv2.putText(image,
                    f'connections: {len(connections)}',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)

        cv2.imwrite(out_path, image)
        print(f'可视化: {out_path}')
