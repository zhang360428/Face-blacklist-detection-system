# face_detection_service.py
import os
import sys
import shutil
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from PIL import Image
import io
from face_model import FaceRecognitionModel
from milvus_client import MilvusClient
import milvus_client
from config import DEVICE, THRESHOLD

# ==================== 配置区域 ====================
# Milvus集合名称配置（推荐在此修改，而不是直接修改milvus_client模块）
MILVUS_COLLECTION_NAME = "face_library_20251230"
milvus_client.COLLECTION_NAME = MILVUS_COLLECTION_NAME

# 上传文件临时存储目录
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 允许的图片格式
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# ==================================================

# 创建FastAPI应用
app = FastAPI(
    title="人脸黑名单检测服务",
    description="""
    检测图片是否在黑名单人脸库中。
    
    提供两种方式：
    1. Web界面：访问根路径 `/` 直接上传图片
    2. API接口：使用 `/detect` (图片路径) 或 `/upload` (文件上传)
    """,
    version="1.0.0"
)

# 请求模型
class ImagePathRequest(BaseModel):
    image_path: str = Field(..., description="图片文件的绝对路径")

# ==================== 新增：Base64请求模型 ====================
class ImageBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64编码的图片数据")
    filename: Optional[str] = Field("base64_image.jpg", description="文件名（可选）")
# ============================================================

# 响应模型
class DetectionResponse(BaseModel):
    status: str = Field(..., description="处理状态: success或error")
    image_path: Optional[str] = Field(None, description="输入的图片路径")
    detected: bool = Field(..., description="是否检测到人脸")
    predicted_in_blacklist: bool = Field(..., description="是否在黑人脸库中")
    matched_person: Optional[str] = Field(None, description="匹配到的人名")
    similarity: float = Field(0.0, description="相似度分数")
    face_id: Optional[str] = Field(None, description="匹配的人脸ID")
    threshold: float = Field(..., description="判定阈值")
    processing_time: Optional[float] = Field(None, description="处理耗时(秒)")

# 全局实例
face_model: Optional[FaceRecognitionModel] = None
milvus_client_instance: Optional[MilvusClient] = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    global face_model, milvus_client_instance
    
    print("="*60)
    print("正在初始化人脸检测服务...")
    
    try:
        # 初始化人脸模型
        face_model = FaceRecognitionModel(device=DEVICE)
        
        # 初始化Milvus客户端（传入自定义集合名称）
        milvus_client_instance = MilvusClient()
        
        # 验证连接
        stats = milvus_client_instance.get_collection_stats()
        
        print(f"服务初始化成功！")
        print(f"Milvus集合: {MILVUS_COLLECTION_NAME}")
        print(f"特征向量数: {stats}")
        print(f"判定阈值: {THRESHOLD}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 服务初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_image(image_data: np.ndarray, filename: str = "uploaded_image") -> DetectionResponse:
    """统一的图片处理逻辑"""
    import time
    start_time = time.time()
    
    # 提取人脸特征
    try:
        # 将numpy数组转换为临时文件（face_model需要文件路径）
        temp_path = UPLOAD_DIR / f"temp_{int(start_time)}_{filename}"
        img = Image.fromarray(image_data)
        
        # 修复：将RGBA模式转换为RGB（JPEG不支持透明度通道）
        # if img.mode == 'RGBA':
        #     img = img.convert('RGB')
        # 转换为RGB（处理透明通道）
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                # 透明背景处理
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert("RGB")
        
        img.save(temp_path)
        
        success, feature, _ = face_model.extract_feature(str(temp_path))
        
        # 清理临时文件
        temp_path.unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"特征提取失败: {str(e)}"
        )
    
    if not success:
        return DetectionResponse(
            status="success",
            image_path=filename,
            detected=False,
            predicted_in_blacklist=False,
            similarity=0.0,
            threshold=THRESHOLD,
            processing_time=round(time.time() - start_time, 3)
        )
    
    # 在Milvus中搜索
    try:
        is_match, person_name, similarity, face_id = milvus_client_instance.search_face(feature)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Milvus搜索失败: {str(e)}"
        )
    
    # 判断是否命中黑名单
    predicted_in_blacklist = bool(is_match and similarity >= THRESHOLD)
    
    return DetectionResponse(
        status="success",
        image_path=filename,
        detected=True,
        predicted_in_blacklist=predicted_in_blacklist,
        matched_person=person_name if is_match else None,
        similarity=float(similarity) if is_match else 0.0,
        face_id=face_id if is_match else None,
        threshold=THRESHOLD,
        processing_time=round(time.time() - start_time, 3)
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_face(request: ImagePathRequest):
    """
    检测单张图片是否在黑人脸库中（通过图片路径）
    
    请求示例:
    ```json
    {
        "image_path": "/path/to/image.jpg"
    }
    ```
    """
    image_path = request.image_path
    
    # 验证文件存在
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_path}")
    
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=400, detail=f"路径不是有效的文件: {image_path}")
    
    # 验证文件格式
    if not Path(image_path.lower()).suffix in VALID_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式。支持: {VALID_IMAGE_EXTENSIONS}"
        )
    
    # 读取图片
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {str(e)}")
    
    return process_image(img_array, Path(image_path).name)

@app.post("/upload", response_model=DetectionResponse)
async def upload_face(file: UploadFile = File(...)):
    """
    上传图片文件进行人脸检测
    
    使用 multipart/form-data 格式上传文件
    """
    # 验证文件类型
    file_ext = Path(file.filename.lower()).suffix
    if file_ext not in VALID_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。支持: {VALID_IMAGE_EXTENSIONS}"
        )
    
    # 读取文件内容
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取上传的图片: {str(e)}")
    
    return process_image(img_array, file.filename)

# ==================== 新增：Base64检测接口 ====================
@app.post("/detect_base64", response_model=DetectionResponse)
async def detect_face_base64(request: ImageBase64Request):
    import base64
    import re
    
    try:
        # 获取base64数据（去掉data URI前缀）
        image_base64 = request.image_base64
        
        # 清理Base64字符串（移除换行符和空格）
        image_base64 = image_base64.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # 处理data URI格式 (data:image/jpeg;base64,)
        if ',' in image_base64:
            match = re.match(r'data:image/[^;]+;base64,(.*)', image_base64)
            if match:
                image_base64 = match.group(1)
        
        # 解码base64（添加错误处理）
        try:
            # 确保字符串长度是4的倍数
            missing_padding = len(image_base64) % 4
            if missing_padding != 0:
                image_base64 += '=' * (4 - missing_padding)
            
            image_data = base64.b64decode(image_base64, validate=True)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Base64解码失败: {str(e)}。请确保提供有效的Base64编码图片数据。"
            )
        
        # 验证解码后的数据不为空
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Base64解码后数据为空"
            )
        
        # 将解码后的数据转换为图片
        try:
            img = Image.open(io.BytesIO(image_data))
            img_array = np.array(img)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"无法解析图片数据: {str(e)}。可能不是有效的图片格式。"
            )
        
        # 验证图片数据有效性
        if img_array is None or img_array.size == 0:
            raise HTTPException(
                status_code=400,
                detail="无效的图片数据"
            )
        
        # 调用统一处理函数
        return process_image(img_array, request.filename)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理base64图片失败: {str(e)}"
        )
# ============================================================

async def detect_face_base64_old(request: ImageBase64Request):
    """
    检测base64编码的图片是否在黑人脸库中
    
    请求示例:
    ```json
    {
        "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
        "filename": "test_image.jpg"
    }
    ```
    或纯base64:
    ```json
    {
        "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
    }
    ```
    """
    import base64
    import re
    
    try:
        # 获取base64数据（去掉data URI前缀）
        image_base64 = request.image_base64
        
        # 处理data URI格式 (data:image/jpeg;base64,)
        if ',' in image_base64:
            # 检查是否为data URI格式
            match = re.match(r'data:image/[^;]+;base64,(.*)', image_base64)
            if match:
                image_base64 = match.group(1)
            else:
                # 如果有逗号但不是data URI格式，取逗号后的部分
                image_base64 = image_base64.split(',')[-1]
        
        # 解码base64
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Base64解码失败: {str(e)}"
            )
        
        # 将解码后的数据转换为图片
        try:
            img = Image.open(io.BytesIO(image_data))
            img_array = np.array(img)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"无法解析图片数据: {str(e)}"
            )
        
        # 验证图片格式
        if img.format.lower() not in ['jpeg', 'jpg', 'png', 'bmp', 'webp']:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图片格式。支持: {VALID_IMAGE_EXTENSIONS}"
            )
        
        # 调用统一处理函数
        return process_image(img_array, request.filename)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理base64图片失败: {str(e)}"
        )
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """提供Web前端界面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>人脸黑名单检测系统</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', 'Microsoft YaHei', system-ui, sans-serif;
                background: linear-gradient(135deg, #f0f5ff 0%, #e6f0ff 100%);
                min-height: 100vh;
                padding: 20px;
                color: #1a2b4d;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
                overflow: hidden;
                backdrop-filter: blur(4px);
                -webkit-backdrop-filter: blur(4px);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            
            .header {
                background: linear-gradient(135deg, #2c5cc5 0%, #3a6bd9 100%);
                color: white;
                padding: 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
                opacity: 0.1;
            }
            
            .header h1 {
                font-size: 2.2em;
                margin-bottom: 10px;
                position: relative;
                z-index: 1;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.85;
                position: relative;
                z-index: 1;
            }
            
            .content {
                padding: 30px;
            }
            
            /* 统计信息区域 - 修改样式防止重叠 */
            .stats {
                background: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                box-shadow: 0 4px 16px rgba(0,0,0,0.03);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }
            
            .stat-item {
                text-align: center;
                padding: 10px;
                background: rgba(255, 255, 255, 0.6);
                border-radius: 8px;
                border: 1px solid rgba(200, 220, 255, 0.4);
            }
            
            .stat-value {
                font-size: 2em;
                font-weight: 700;
                color: #2c5cc5;
                text-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .stat-label {
                color: #5a6d8a;
                margin-top: 8px;
                font-weight: 500;
                font-size: 0.9em;
            }
            
            /* Milvus集合信息 - 单独一行显示 */
            .collection-info {
                background: rgba(230, 240, 255, 0.6);
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #2c5cc5;
                margin-top: 15px;
            }
            
            .collection-info-label {
                font-size: 0.85em;
                color: #5a6d8a;
                margin-bottom: 5px;
                font-weight: 500;
            }
            
            .collection-info-value {
                font-size: 1em;
                color: #1a2b4d;
                font-weight: 600;
                word-break: break-all;
                font-family: 'Courier New', monospace;
            }
            
            .upload-area {
                border: 2px dashed #c2d6ff;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                background: #f8fbff;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .upload-area:hover {
                border-color: #5b8cff;
                background: #f0f6ff;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(44, 92, 197, 0.1);
            }
            
            .upload-area.dragover {
                border-color: #2c5cc5;
                background: #e6f0ff;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 3.5em;
                color: #5b8cff;
                margin-bottom: 20px;
                opacity: 0.8;
            }
            
            .file-input {
                display: none;
            }
            
            .upload-btn {
                background: linear-gradient(135deg, #2c5cc5 0%, #3a6bd9 100%);
                color: white;
                padding: 14px 36px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1.1em;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(44, 92, 197, 0.25);
                position: relative;
                overflow: hidden;
                font-weight: 500;
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(44, 92, 197, 0.35);
            }
            
            .upload-btn:active {
                transform: translateY(0);
                box-shadow: 0 2px 8px rgba(44, 92, 197, 0.25);
            }
            
            .upload-btn::after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(rgba(255,255,255,0.13), rgba(255,255,255,0));
                transform: rotate(30deg);
            }
            
            .preview-area {
                display: none;
                margin: 30px 0;
            }
            
            .preview-card {
                display: inline-block;
                padding: 10px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #eef4ff;
            }
            
            .preview-image {
                max-width: 100%;
                max-height: 320px;
                border-radius: 8px;
                display: block;
            }
            
            .file-info {
                margin-top: 12px;
                font-size: 0.9em;
                color: #5a6d8a;
                text-align: center;
                font-weight: 500;
            }
            
            .result-area {
                margin-top: 30px;
            }
            
            .result-card {
                padding: 25px;
                border-radius: 12px;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.8);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            }
            
            .result-success {
                background: rgba(230, 245, 230, 0.7);
                border: 1px solid rgba(74, 181, 74, 0.3);
            }
            
            .result-warning {
                background: rgba(255, 243, 224, 0.7);
                border: 1px solid rgba(255, 179, 71, 0.3);
            }
            
            .result-danger {
                background: rgba(255, 235, 238, 0.7);
                border: 1px solid rgba(244, 67, 54, 0.3);
            }
            
            .result-info {
                background: rgba(227, 242, 253, 0.7);
                border: 1px solid rgba(66, 165, 245, 0.3);
            }
            
            .result-title {
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .result-details {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 15px;
            }
            
            .detail-item {
                padding: 12px 0;
                border-bottom: 1px dashed #e0e9ff;
            }
            
            .detail-label {
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 5px;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .detail-value {
                font-size: 1.15em;
                color: #1a2b4d;
                font-weight: 500;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 30px;
            }
            
            .spinner {
                width: 50px;
                height: 50px;
                border: 4px solid #e6f0ff;
                border-top: 4px solid #2c5cc5;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
                box-shadow: 0 0 10px rgba(44, 92, 197, 0.2);
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error-message {
                background: rgba(255, 235, 238, 0.8);
                color: #c62828;
                padding: 16px;
                border-radius: 8px;
                margin: 20px 0;
                display: none;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
                border: 1px solid rgba(244, 67, 54, 0.15);
            }
            
            .success-message {
                background: rgba(232, 245, 232, 0.8);
                color: #2e7d32;
                padding: 16px;
                border-radius: 8px;
                margin: 20px 0;
                display: none;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
                border: 1px solid rgba(74, 181, 74, 0.15);
            }
            
            .progress-bar {
                width: 100%;
                height: 4px;
                background: #e6f0ff;
                border-radius: 2px;
                overflow: hidden;
                margin: 15px 0;
                display: none;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #2c5cc5, #4a90e2);
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 2px;
            }
            
            @media (max-width: 600px) {
                .container {
                    margin: 10px;
                    border-radius: 12px;
                }
                
                .header h1 {
                    font-size: 1.8em;
                }
                
                .content {
                    padding: 20px;
                }
                
                .upload-area {
                    padding: 25px 20px;
                }
                
                .upload-btn {
                    padding: 12px 24px;
                    font-size: 1em;
                }
                
                .preview-image {
                    max-height: 250px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ 人脸黑名单检测系统</h1>
                <p>基于深度学习的人脸识别与匹配服务</p>
            </div>
            
            <div class="content">
                <!-- 统计信息区域 - 修改布局 -->
                <div class="stats" id="stats">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="faceCount">-</div>
                            <div class="stat-label">黑名单人脸数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="threshold">-</div>
                            <div class="stat-label">判定阈值</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="device">-</div>
                            <div class="stat-label">计算设备</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="serviceStatus">-</div>
                            <div class="stat-label">服务状态</div>
                        </div>
                    </div>
                    <!-- Milvus集合名称单独显示 -->
                    <div class="collection-info">
                        <div class="collection-info-label">Milvus集合名称</div>
                        <div class="collection-info-value" id="collectionName">-</div>
                    </div>
                </div>
                
                <div class="error-message" id="errorMessage"></div>
                <div class="success-message" id="successMessage"></div>
                
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📷</div>
                    <h3 style="margin-bottom: 15px; color: #34495e;">上传图片进行检测</h3>
                    <p style="color: #7f8c8d; margin-bottom: 20px;">支持 JPG, PNG, BMP 格式</p>
                    <input type="file" id="fileInput" class="file-input" accept="image/*">
                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                        选择图片
                    </button>
                    <div class="progress-bar" id="progressBar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                </div>
                
                <div class="preview-area" id="previewArea">
                    <div class="preview-card">
                        <img id="previewImage" class="preview-image" alt="预览图片">
                        <div class="file-info" id="fileInfo"></div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>正在分析图片，请稍候...</p>
                </div>
                
                <div class="result-area" id="resultArea"></div>
            </div>
        </div>
        
        <script>
            let uploadedFile = null;
            
            // 页面加载时获取统计信息
            window.onload = async function() {
                await loadStats();
            };
            
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    
                    document.getElementById('faceCount').textContent = data.total_faces || '-';
                    document.getElementById('threshold').textContent = data.threshold ? (data.threshold * 100).toFixed(1) + '%' : '-';
                    document.getElementById('device').textContent = data.device || 'CPU';
                    document.getElementById('collectionName').textContent = data.collection_name || '-';
                    document.getElementById('serviceStatus').textContent = data.status === 'ok' ? '正常' : '异常';
                } catch (error) {
                    console.error('获取统计信息失败:', error);
                }
            }
            
            // 文件选择处理
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    uploadedFile = file;
                    previewFile(file);
                }
            });
            
            // 拖拽上传
            const uploadArea = document.getElementById('uploadArea');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.type.startsWith('image/')) {
                        uploadedFile = file;
                        previewFile(file);
                    } else {
                        showError('请选择图片文件！');
                    }
                }
            });
            
            // 预览文件
            function previewFile(file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const previewArea = document.getElementById('previewArea');
                    const previewImage = document.getElementById('previewImage');
                    const fileInfo = document.getElementById('fileInfo');
                    
                    previewImage.src = e.target.result;
                    fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                    previewArea.style.display = 'block';
                    
                    // 自动开始检测
                    uploadAndDetect(file);
                };
                
                reader.readAsDataURL(file);
            }
            
            // 上传并检测
            async function uploadAndDetect(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                // 显示加载状态
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultArea').innerHTML = '';
                hideError();
                hideSuccess();
                
                // 显示进度条
                showProgress();
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        displayResult(result);
                        hideProgress();
                        
                        // 显示成功消息
                        if (result.processing_time) {
                            showSuccess(`检测完成！耗时 ${result.processing_time} 秒`);
                        }
                    } else {
                        const error = await response.json();
                        throw new Error(error.detail || '检测失败');
                    }
                } catch (error) {
                    showError(error.message);
                    hideProgress();
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            // 显示进度条
            function showProgress() {
                const progressBar = document.getElementById('progressBar');
                const progressFill = document.getElementById('progressFill');
                
                progressBar.style.display = 'block';
                progressFill.style.width = '0%';
                
                // 模拟进度
                let progress = 0;
                const interval = setInterval(() => {
                    progress += Math.random() * 25;
                    if (progress > 90) progress = 90;
                    progressFill.style.width = progress + '%';
                }, 200);
                
                window.progressInterval = interval;
            }
            
            // 隐藏进度条
            function hideProgress() {
                if (window.progressInterval) {
                    clearInterval(window.progressInterval);
                }
                const progressBar = document.getElementById('progressBar');
                const progressFill = document.getElementById('progressFill');
                
                progressFill.style.width = '100%';
                setTimeout(() => {
                    progressBar.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 300);
            }
            
            // 显示结果
            function displayResult(result) {
                const resultArea = document.getElementById('resultArea');
                
                let resultClass = 'result-info';
                let icon = 'ℹ️';
                let title = '检测结果';
                
                if (result.status === 'error') {
                    resultClass = 'result-danger';
                    icon = '❌';
                    title = '检测失败';
                } else if (!result.detected) {
                    resultClass = 'result-warning';
                    icon = '⚠️';
                    title = '未检测到人脸';
                } else if (result.predicted_in_blacklist) {
                    resultClass = 'result-danger';
                    icon = '🚨';
                    title = '⚠️ 命中黑名单！';
                } else {
                    resultClass = 'result-success';
                    icon = '✅';
                    title = '未在黑名单中';
                }
                
                resultArea.innerHTML = `
                    <div class="result-card ${resultClass}">
                        <div class="result-title">${icon} ${title}</div>
                        <div class="result-details">
                            ${result.detected ? `
                                <div class="detail-item">
                                    <div class="detail-label">检测状态</div>
                                    <div class="detail-value">✅ 人脸检测成功</div>
                                </div>
                                ${result.predicted_in_blacklist ? `
                                    <div class="detail-item">
                                        <div class="detail-label">匹配人员</div>
                                        <div class="detail-value" style="color: #e74c3c; font-weight: bold;">${result.matched_person || '未知'}</div>
                                    </div>
                                    <div class="detail-item">
                                        <div class="detail-label">相似度</div>
                                        <div class="detail-value" style="color: #e74c3c; font-weight: bold;">${(result.similarity * 100).toFixed(2)}%</div>
                                    </div>
                                    <div class="detail-item">
                                        <div class="detail-label">人脸ID</div>
                                        <div class="detail-value">${result.face_id || '-'}</div>
                                    </div>
                                ` : `
                                    <div class="detail-item">
                                        <div class="detail-label">匹配结果</div>
                                        <div class="detail-value" style="color: #27ae60;">✅ 未匹配到黑名单</div>
                                    </div>
                                    <div class="detail-item">
                                        <div class="detail-label">最高相似度</div>
                                        <div class="detail-value">${(result.similarity * 100).toFixed(2)}%</div>
                                    </div>
                                `}
                                <div class="detail-item">
                                    <div class="detail-label">判定阈值</div>
                                    <div class="detail-value">${(result.threshold * 100).toFixed(2)}%</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">处理时间</div>
                                    <div class="detail-value">${result.processing_time || '-'} 秒</div>
                                </div>
                            ` : `
                                <div class="detail-item">
                                    <div class="detail-label">检测状态</div>
                                    <div class="detail-value" style="color: #f39c12;">⚠️ 未检测到人脸</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">建议</div>
                                    <div class="detail-value">请上传正面清晰的人脸照片</div>
                                </div>
                            `}
                        </div>
                    </div>
                `;
            }
            
            // 显示/隐藏错误信息
            function showError(message) {
                const errorElement = document.getElementById('errorMessage');
                errorElement.textContent = message;
                errorElement.style.display = 'block';
                
                setTimeout(() => {
                    hideError();
                }, 5000);
            }
            
            function hideError() {
                document.getElementById('errorMessage').style.display = 'none';
            }
            
            function showSuccess(message) {
                const successElement = document.getElementById('successMessage');
                successElement.textContent = message;
                successElement.style.display = 'block';
                
                setTimeout(() => {
                    hideSuccess();
                }, 3000);
            }
            
            function hideSuccess() {
                document.getElementById('successMessage').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """


@app.get("/stats")
async def get_stats():
    """获取人脸库统计信息（包含设备信息）"""
    try:
        count = milvus_client_instance.get_collection_stats()
        return {
            "collection_name": MILVUS_COLLECTION_NAME,
            "total_faces": count,
            "status": "ok",
            "threshold": THRESHOLD,
            "device": DEVICE,
            "service": "人脸黑名单检测服务"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "face-detection-service",
        "collection": MILVUS_COLLECTION_NAME,
        "device": DEVICE
    }

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "face_detection_service:app",
        host="0.0.0.0",
        port=9876,
        reload=False,
        log_level="info"
    )
