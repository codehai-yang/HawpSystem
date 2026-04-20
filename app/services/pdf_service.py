import os
from cmath import rect

# 配置 PaddleX 模型目录（用于 doc_orientation 和 unwarp 模型） 告诉PaddleX 从哪个目录加载所需的模型文件
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# paddlex模型存储路径
os.environ['PADDLEX_HOME'] = os.path.join(_project_root, 'models')

from ultralytics import YOLO
import cv2
from paddlex import create_model
import torch
import base64
import numpy as np

# 单例模型实例缓存
_yolo_model = None
_rec_model = None


def get_yolo_model():
    """获取 YOLO 模型单例"""
    global _yolo_model
    if _yolo_model is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            r"F:\office\pythonProjects\SystemVision-原理图识别\yolo\yolov8_train\runs\yolov8_train_20260418_134750\exp\weights\best.pt"
        )
        _yolo_model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO model initialized, using device: {device}")
        _yolo_model = _yolo_model.to(device)
    return _yolo_model

def get_rec_model():
    """获取 PaddleX rec 识别模型单例"""
    global _rec_model
    if _rec_model is None:
        _rec_model = init_rec_model()
        print("PaddleX rec model initialized (rec-only mode)")
    return _rec_model
# ── 主处理函数 ────────────────────────────────────────────────────────────────

def process_single_page_image(image_data_base64: str, page_number: int = 1):
    """
    处理单页图像的 OCR 识别。

    Args:
        image_data_base64: base64 编码的图像数据（可能带 data URL 前缀）
        page_number: 页码（用于日志）

    Returns:
        识别结果列表，格式与原版一致
    """
    # 去除可能的 data URL 前缀
    if ',' in image_data_base64:
        image_data_base64 = image_data_base64.split(',')[1]

    # 解码 base64 图像
    image_bytes = base64.b64decode(image_data_base64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("无法解码图像数据")

    h, w = img.shape[:2]
    print(f"Processing single page {page_number}, image shape: {img.shape}")

    # 获取模型单例
    yolo_model = get_yolo_model()
    rec_model  = get_rec_model()

    # ── YOLO 检测 ─────────────────────────────────────────────────────────────
    results = yolo_model(img, iou=0.1, conf=0.05,imgsz=5120,rect=True,augment=True)
    print(f'[DEBUG] YOLO 输入图像尺寸: {w} x {h} (宽x高)')

    detect_results = []
    for result in results:
        image_height, image_width = result.orig_shape
        print(f'[DEBUG] YOLO orig_shape: {image_width} x {image_height} (宽x高)')

        if result.orig_shape != img.shape[:2]:
            print(f'[WARNING] YOLO 内部对图像进行了预处理！'
                  f' 输入={img.shape[:2]}, orig_shape={result.orig_shape}')

        for box in result.boxes:
            cls       = int(box.cls[0])
            cls_name  = yolo_model.names[cls]
            conf      = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            if len(detect_results) < 3:
                print(f'[DEBUG] 检测框 {len(detect_results)}: class={cls_name}, '
                      f'box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), '
                      f'size={x2-x1:.0f}x{y2-y1:.0f}')

            detect_results.append({
                "class_id":   cls,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": {
                    "x_min":  x1, "y_min": y1,
                    "x_max":  x2, "y_max": y2,
                    "width":  x2 - x1, "height": y2 - y1,
                },
                "source": "yolo",
            })

    print(f"YOLO detected {len(detect_results)} objects on page {page_number}")

    # ── 筛选目标类别 ────────────────────────────────────────────────────────────
    target_classes  = ['signalName', 'device', 'power', 'ground']
    target_objects  = [obj for obj in detect_results if obj.get('class_name') in target_classes]
    print(f"Found {len(target_objects)} target objects on page {page_number}")

    # ── 收集 signalName ROI 准备批量 OCR ─────────────────────────────────────
    valid_objects = []
    rois = []

    for i, obj in enumerate(target_objects):
        if obj.get('class_name') != 'signalName':
            continue
        bbox_dict = obj.get('bbox', {})

        try:
            x_min = int(bbox_dict['x_min'])
            y_min = int(bbox_dict['y_min'])
            x_max = int(bbox_dict['x_max'])
            y_max = int(bbox_dict['y_max'])
        except (KeyError, ValueError) as e:
            print(f"警告：目标 {i + 1} 的坐标无效，跳过: {e}")
            continue

        x_min = max(0, min(x_min, w))
        y_min = max(0, min(y_min, h))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))

        if x_max <= x_min or y_max <= y_min:
            print(f"警告：目标 {i + 1} 的区域无效，跳过")
            continue

        points = [
            {"x": x_min, "y": y_min},
            {"x": x_max, "y": y_min},
            {"x": x_max, "y": y_max},
            {"x": x_min, "y": y_max},
        ]
        roi = img[y_min:y_max, x_min:x_max]

        valid_objects.append({"label": obj.get('class_name'), "points": points, "index": i})
        rois.append(roi)

    print(f"Collected {len(rois)} valid ROIs for batch OCR processing")

    # ── PaddleX rec 批量 OCR ─────────────────────────────────────────
    objects = []
    if rois:
        print(f"Starting batch rec recognition for {len(rois)} ROIs")
        batch_size = min(6, len(rois))

        try:
            rec_results_generator = rec_model.predict(rois, batch_size=batch_size)
            rec_results_list = list(rec_results_generator)
            print(f"Batch rec completed, got {len(rec_results_list)} results")

            for idx, (obj_meta, rec_result) in enumerate(zip(valid_objects, rec_results_list)):
                text = ""
                try:
                    text = rec_result.get('rec_text', '')
                except Exception as e:
                    print(f"Warning: Failed to extract text for object {idx + 1}: {e}")

                objects.append({
                    "label":  obj_meta["label"],
                    "points": obj_meta["points"],
                    "text":   text,
                })
                print(f"Processed object {idx + 1}/{len(valid_objects)}: {obj_meta['label']} - '{text}'")

        except Exception as e:
            print(f"Warning: Batch OCR failed: {e}，回退到逐个处理...")
            for idx, (obj_meta, roi) in enumerate(zip(valid_objects, rois)):
                text = ""
                try:
                    rec_results = list(rec_model.predict(roi))
                    if rec_results:
                        text = rec_results[0].get('rec_text', '')
                except Exception as inner_e:
                    print(f"Warning: OCR failed for object {idx + 1}: {inner_e}")

                objects.append({
                    "label":  obj_meta["label"],
                    "points": obj_meta["points"],
                    "text":   text,
                })

        print(f"Batch rec recognition completed for {len(rois)} ROIs")

    # ── 非 signalName 目标加入结果（不带 OCR 文本）────────────────────────────
    for obj in target_objects:
        if obj.get('class_name') == 'signalName':
            continue
        bbox_dict = obj.get('bbox', {})
        try:
            x_min = int(bbox_dict['x_min'])
            y_min = int(bbox_dict['y_min'])
            x_max = int(bbox_dict['x_max'])
            y_max = int(bbox_dict['y_max'])
        except (KeyError, ValueError):
            continue

        points = [
            {"x": x_min, "y": y_min},
            {"x": x_max, "y": y_min},
            {"x": x_max, "y": y_max},
            {"x": x_min, "y": y_max},
        ]
        objects.append({
            "label":  obj.get('class_name'),
            "points": points,
            "text":   "",
        })

    print(f"Total objects returned: {len(objects)} "
          f"(signalName with OCR: {len(rois)}, others: {len(objects) - len(rois)})")

    return objects


# ── PaddleX rec 模型初始化 ─────────────────────────────────────────────────────

def init_rec_model():
    """初始化 PaddleX rec 识别模型（仅识别，不检测）"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir   = os.path.join(project_root, 'models', 'official_models')

    use_gpu = torch.cuda.is_available()
    device  = 'gpu' if use_gpu else 'cpu'
    print(f"PaddleX rec model is using device: {device}")

    model_name = 'PP-OCRv5_mobile_rec'
    rec_model  = create_model(
        model_name=model_name,
        model_dir=os.path.join(models_dir, model_name),
        device=device,
    )
    return rec_model
