"""
HAWP 线段检测服务
"""
import os
import cv2
import math
import torch
import numpy as np
from app.core.config import Config


class HawpDetector:
    """HAWP 线段检测器"""

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """加载 HAWP 模型"""
        from hawp.fsl.config import cfg as model_config
        from hawp.fsl.model.build import build_model

        model_config.merge_from_file(Config.CFG_PATH)
        model = build_model(model_config)
        model = model.eval().to(Config.DEVICE)

        state = torch.load(Config.CKPT_PATH, map_location='cpu')
        if 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=False)

        print(f'HAWP 加载成功: {os.path.basename(Config.CKPT_PATH)}')
        return model

    def _predict_tile(self, tile_bgr):
        """预测单个瓦片"""
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        h, w = tile_rgb.shape[:2]

        resized = cv2.resize(tile_rgb, (Config.INFER_SIZE, Config.INFER_SIZE))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1)[None].to(Config.DEVICE)
        meta = [{'width': Config.INFER_SIZE, 'height': Config.INFER_SIZE, 'filename': ''}]

        try:
            with torch.no_grad():
                output, _ = self.model(tensor, meta)
        except Exception:
            return []

        lines = output['lines_pred'].cpu().numpy()
        scores = output['lines_score'].cpu().numpy().flatten()

        sx, sy = w / Config.INFER_SIZE, h / Config.INFER_SIZE
        results = []

        for line, score in zip(lines, scores):
            if score < Config.HAWP_THRESHOLD:
                continue
            results.append({
                'p1': (float(line[0]) * sx, float(line[1]) * sy),
                'p2': (float(line[2]) * sx, float(line[3]) * sy),
                'score': float(score)
            })

        return results

    def detect_lines(self, image_path):
        """
        检测图像中的所有线段（滑动窗口推理）

        Returns:
            tuple: (hawp_lines, ori_w, ori_h)
        """
        image = cv2.imread(image_path)
        ori_h, ori_w = image.shape[:2]

        stride = Config.TILE_SIZE - Config.OVERLAP
        n_cols = math.ceil((ori_w - Config.OVERLAP) / stride)
        n_rows = math.ceil((ori_h - Config.OVERLAP) / stride)
        total = n_rows * n_cols

        all_lines = []
        count = 0

        for row in range(n_rows):
            for col in range(n_cols):
                count += 1
                x1 = int(col * stride)
                y1 = int(row * stride)
                x2 = min(int(x1 + Config.TILE_SIZE), ori_w)
                y2 = min(int(y1 + Config.TILE_SIZE), ori_h)

                tile = image[y1:y2, x1:x2]

                for line in self._predict_tile(tile):
                    all_lines.append({
                        'p1': (line['p1'][0] + x1, line['p1'][1] + y1),
                        'p2': (line['p2'][0] + x1, line['p2'][1] + y1),
                        'score': line['score']
                    })

                print(f'  HAWP [{count}/{total}]', end='\r')

        print(f'\nHAWP 检测到 {len(all_lines)} 条线段')
        return all_lines, ori_w, ori_h
