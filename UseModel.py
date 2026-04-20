"""
predict_sliding_window.py
把高分辨率原理图切成小块分别推理，再把结果拼回原图坐标系。
不需要重新训练，直接用现有 512 训练的权重。

用法：
  python predict_sliding_window.py --img your_image.jpg
  python predict_sliding_window.py --img_dir ./images --out_dir ./results
"""

import os
import cv2
import torch
import argparse
import math

from hawp.fsl.config import cfg as model_config
from hawp.fsl.model.build import build_model



# ── 配置区 ──────────────────────────────────────────
CKPT_PATH  = r'/best.pth'
CFG_PATH   = r'/hawp\fsl\config\hawpv2.yaml'
DEVICE     = 'cuda'
THRESHOLD  = 0.3   # 置信度阈值
INFER_SIZE = 512    # 每个小块推理的分辨率，和训练一致
TILE_SIZE  = 1024   # 裁剪块大小（原图像素），越大保留细节越多
OVERLAP    = 128    # 相邻块的重叠区域，避免边界漏检
# ────────────────────────────────────────────────────


def load_model(ckpt_path, cfg_path, device='cuda'):
    model_config.merge_from_file(cfg_path)
    model = build_model(model_config)
    model = model.eval().to(device)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    print(f'模型加载成功: {os.path.basename(ckpt_path)}')
    return model


def predict_tile(model, tile_bgr, device, threshold, infer_size):
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    h, w = tile_rgb.shape[:2]

    tile_resized = cv2.resize(tile_rgb, (infer_size, infer_size))
    tensor = torch.from_numpy(tile_resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1)[None].to(device)

    meta = [{'width': infer_size, 'height': infer_size, 'filename': ''}]

    try:                                          # ← 加这里
        with torch.no_grad():
            output, _ = model(tensor, meta)
    except Exception as e:                        # ← 捕获空块报错
        return []                                 # ← 直接返回空列表

    lines  = output['lines_pred'].cpu().numpy()
    scores = output['lines_score'].cpu().numpy().flatten()

    scale_x = w / infer_size
    scale_y = h / infer_size

    results = []
    for line, score in zip(lines, scores):
        if score < threshold:
            continue
        results.append({
            'p1':    (float(line[0]) * scale_x, float(line[1]) * scale_y),
            'p2':    (float(line[2]) * scale_x, float(line[3]) * scale_y),
            'score': float(score)
        })
    return results


def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def deduplicate_lines(all_lines, dist_threshold=8.0):
    """
    去除重叠区域产生的重复线段。
    两条线段的四个端点互相都很近，视为重复，保留 score 更高的那条。
    """
    if not all_lines:
        return []

    all_lines = sorted(all_lines, key=lambda x: -x['score'])
    kept = []
    for line in all_lines:
        duplicate = False
        for k in kept:
            # 正向匹配
            d1 = (dist(line['p1'], k['p1']) + dist(line['p2'], k['p2'])) / 2
            # 反向匹配
            d2 = (dist(line['p1'], k['p2']) + dist(line['p2'], k['p1'])) / 2
            if min(d1, d2) < dist_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def predict_large_image(model, image_path, device, threshold,
                        infer_size, tile_size, overlap):
    """
    滑动窗口推理大图。
    把原图切成 tile_size×tile_size 的块（有重叠），
    分别推理后把坐标映射回原图坐标系，最后去重。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'找不到图片: {image_path}')

    ori_h, ori_w = image.shape[:2]
    print(f'原图尺寸: {ori_w}×{ori_h}')

    stride = tile_size - overlap  # 滑动步长

    # 计算需要多少行列的块
    n_cols = math.ceil((ori_w - overlap) / stride)
    n_rows = math.ceil((ori_h - overlap) / stride)
    total  = n_rows * n_cols
    print(f'切块方案: {n_cols}列×{n_rows}行 = {total} 块')

    all_lines = []
    count = 0

    for row in range(n_rows):
        for col in range(n_cols):
            count += 1
            # 计算裁剪坐标
            x1 = col * stride
            y1 = row * stride
            x2 = min(x1 + tile_size, ori_w)
            y2 = min(y1 + tile_size, ori_h)

            tile = image[y1:y2, x1:x2]

            # 推理
            tile_lines = predict_tile(
                model, tile, device, threshold, infer_size)

            # 坐标映射回原图
            for line in tile_lines:
                all_lines.append({
                    'p1':    (line['p1'][0] + x1, line['p1'][1] + y1),
                    'p2':    (line['p2'][0] + x1, line['p2'][1] + y1),
                    'score': line['score']
                })

            print(f'  [{count}/{total}] 块({row},{col}) '
                  f'区域({x1},{y1})-({x2},{y2}): '
                  f'{len(tile_lines)} 条线段', end='\r')

    print(f'\n合并前共 {len(all_lines)} 条线段，正在去重...')

    # 去重
    final_lines = deduplicate_lines(all_lines, dist_threshold=10.0)
    print(f'去重后 {len(final_lines)} 条线段')

    return final_lines, ori_w, ori_h


def draw_lines(image_path, lines, out_path,
               line_color=(0, 200, 80),
               point_color=(220, 50, 50),
               line_thickness=2,
               point_radius=5,
               show_score=False):
    image = cv2.imread(image_path)
    for line in lines:
        x1, y1 = int(round(line['p1'][0])), int(round(line['p1'][1]))
        x2, y2 = int(round(line['p2'][0])), int(round(line['p2'][1]))
        cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)
        cv2.circle(image, (x1, y1), point_radius, point_color, -1)
        cv2.circle(image, (x2, y2), point_radius, point_color, -1)
        if show_score:
            mx, my = (x1+x2)//2, (y1+y2)//2
            cv2.putText(image, f'{line["score"]:.2f}',
                        (mx, my), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100, 100, 255), 1)

    cv2.putText(image, f'lines: {len(lines)}',
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 200), 2)
    cv2.imwrite(out_path, image)
    print(f'结果保存: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',       default=None)
    parser.add_argument('--img_dir',   default=None)
    parser.add_argument('--out_dir',   default='./results')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--tile_size', type=int,   default=TILE_SIZE)
    parser.add_argument('--overlap',   type=int,   default=OVERLAP)
    parser.add_argument('--show_score', action='store_true')
    args = parser.parse_args()

    model = load_model(CKPT_PATH, CFG_PATH, DEVICE)
    os.makedirs(args.out_dir, exist_ok=True)

    def run(img_path):
        lines, _, _ = predict_large_image(
            model, img_path, DEVICE,
            args.threshold, INFER_SIZE,
            args.tile_size, args.overlap)
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.out_dir, 'result_' + fname)
        draw_lines(img_path, lines, out_path,
                   show_score=args.show_score)

    if args.img:
        run(args.img)
    elif args.img_dir:
        imgs = [f for f in os.listdir(args.img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for i, f in enumerate(imgs):
            print(f'\n[{i+1}/{len(imgs)}] {f}')
            run(os.path.join(args.img_dir, f))
    else:
        print('请指定 --img 或 --img_dir')