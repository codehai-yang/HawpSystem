# app/models/model.py
from pydantic import BaseModel, field_validator
from typing import List, Optional

class Point(BaseModel):
    x: int
    y: int

    class Config:
        extra = "ignore"

class ObjectItem(BaseModel):
    label: str
    points: List[Point]
    text: str
    originalText: str = ""  # 给默认值，避免验证错误

    class Config:
        extra = "ignore"

class ImageData(BaseModel):
    objects: List[ObjectItem]

    class Config:
        extra = "ignore"

class OcrResult(BaseModel):
    fileName: str
    image_count: int
    images: List[ImageData]
    workDir: str = ""

    class Config:
        extra = "ignore"

    @field_validator('fileName')
    @classmethod
    def filename_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('文件名不能为空')
        return v

    @field_validator('image_count')
    @classmethod
    def image_count_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('图像数量必须大于0')
        return v


class PageImageRequest(BaseModel):
    """单页图像OCR请求"""
    image_data: str  # base64编码的图像数据
    page_number: int = 1  # 页码（从1开始）

    class Config:
        extra = "ignore"


class PageOcrResult(BaseModel):
    """单页OCR结果"""
    page_number: int
    objects: List[ObjectItem]

    class Config:
        extra = "ignore"
