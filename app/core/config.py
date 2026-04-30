import os
import sys
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DEBUG: bool = os.getenv("DEBUG", False)
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "BackendOCR")

settings = Settings()



class Config:
    """HAWP + YOLO 系统配置"""

    # HAWP 模型配置
    HAWP_ROOT = r'F:\office\pythonProjects\HAWP\hawp'
    CKPT_PATH = r'F:\office\pythonProjects\YOLOandOCR\AutoLogic\BackendOCR\hawpv2-edb9b23f.pth'
    CFG_PATH = r'F:\office\pythonProjects\YOLOandOCR\AutoLogic\BackendOCR\hawpv2.yaml'
    DEVICE = 'cuda'
    HAWP_THRESHOLD = 0.05

    # 推理参数
    INFER_SIZE = 512
    TILE_SIZE = 1024
    OVERLAP = 0.5

    # 图构建参数
    MERGE_TH = 13          # 端点合并距离（像素）

    # 入口点检测参数
    EDGE_TOL = 16         # 设备框边缘容差（像素）
    BOX_EXPAND = 25        # YOLO框向外扩展的像素距离

    # OCR 匹配参数
    OCR_DIST = 50         # OCR文字匹配距离（像素）

    # 设备框合并参数
    WIDTH_RATIO_THRESHOLD = 0.3
    VERTICAL_GAP_THRESHOLD = 4000
    TOP_EXTEND = 350
    BOTTOM_EXTEND = 300
    MAX_ATTEMPTS = 3

    # 可视化参数
    TOUCH_THRESHOLD = 5   # 设备框触碰阈值


# 确保 HAWP 路径在 sys.path 中
if Config.HAWP_ROOT not in sys.path:
    sys.path.insert(0, Config.HAWP_ROOT)