import os
# 禁用 Ultralytics 在线检查（必须在导入 YOLO 之前设置），确保yolo模型加载使用本地资源，避免在加载yolo模型时出现网络请求
os.environ['YOLO_OFFLINE'] = '1'  # 启用离线模式，禁用yolo库进行在线检查，更新和云服务调用，在没有网络连接或网络受限的环境中正常运行
os.environ['ULTRALYTICS_SETTINGS_SYNC'] = 'false'  # 禁用设置同步，阻止库将配置设置同步到云端服务器，保护隐私和避免不必要的网络通信
os.environ['YOLO_AUTOINSTALL'] = 'false'  # 禁用自动安装，防止库自动下载和安装缺失的依赖项，在生产环境中避免意外的包安装操作
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # 跳过检查模型服务这行

from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.api.routes.pdf_routes import router as pdf_router
import uvicorn
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()
app.include_router(pdf_router)


# 前端打包后的静态文件目录
frontend_dist = Path(__file__).parent / "static"

# 如果前端已打包，则 serve 静态文件
if frontend_dist.exists():
    # 挂载静态资源 (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
    
    # 根路由返回 index.html
    @app.get("/")
    async def serve_frontend():
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=49989, log_level="info")

