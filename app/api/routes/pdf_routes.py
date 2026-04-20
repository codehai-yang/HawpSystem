# app/api/routes/pdf_routes.py
from fastapi import APIRouter, HTTPException
import logging

router = APIRouter(prefix="/pdf", tags=["pdf"])

from app.services.pdf_service import process_single_page_image
from app.models.model import OcrResult, PageImageRequest

@router.post("/ocr-page")
async def ocr_single_page(request: PageImageRequest):
    """处理单页图像的OCR识别"""
    try:
        logging.info(f"收到单页OCR请求, 页码: {request.page_number}")

        result = process_single_page_image(
            request.image_data,
            request.page_number
        )

        logging.info(f"单页OCR完成, 识别到 {len(result)} 个目标")

        return {
            "page_number": request.page_number,
            "objects": result
        }

        #二值化图像
    except Exception as e:
        import traceback
        logging.error(f"单页OCR处理失败详情:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"单页OCR处理失败: {str(e)}")