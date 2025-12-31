import os
from pathlib import Path
import cv2
import albumentations as A
from PIL import Image
import numpy as np

def create_albumentations_pipeline():
    """
    使用albumentations创建专业的人脸数据增强pipeline
    """
    # 定义数据增强组合
    transform = A.Compose([
        # 水平翻转（50%概率）
        A.HorizontalFlip(p=0.5),
        
        # 轻微旋转（-180度到180度）
        A.Rotate(limit=180, p=0.5),
        
        # 随机亮度对比度调整
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # ±20%
            contrast_limit=0.2,    # ±20%
            p=0.5
        ),
        
        # 随机缩放和平移
        A.ShiftScaleRotate(
            shift_limit=0.1,   # ±10%平移
            scale_limit=0.1,   # ±10%缩放
            rotate_limit=0,    # 不旋转（上面已有）
            border_mode=cv2.BORDER_CONSTANT,  # 常数填充
            value=0,  # 填充黑色
            p=0.5
        ),
        
        # 轻微高斯模糊
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        
        # 随机锐化
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.3),
        
        # 色度、饱和度、亮度调整
        A.HueSaturationValue(
            hue_shift_limit=10,       # ±10度色相
            sat_shift_limit=20,       # ±20%饱和度
            val_shift_limit=20,       # ±20%明度
            p=0.3
        ),
        
        # 通道随机打乱（模拟不同光照）
        A.ChannelShuffle(p=0.1),
        
        # 添加轻微噪声（模拟低质量摄像头）
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        
        # 色彩抖动
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    ])
    
    return transform

def augment_face_images_albumentations(src_dir, dst_dir):
    """
    使用albumentations对指定目录中的人脸图片进行数据增强
    
    Args:
        src_dir: 源图片目录
        dst_dir: 目标图片目录（增强后的图片）
    """
    # 创建目标目录
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图片文件
    src_path = Path(src_dir)
    image_files = [
        f for f in src_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 创建数据增强pipeline
    transform = create_albumentations_pipeline()
    
    # 处理每张图片
    for i, image_file in enumerate(image_files, 1):
        try:
            # 使用OpenCV读取图片
            # albumentations与OpenCV配合更好
            image = cv2.imread(str(image_file))
            
            if image is None:
                print(f"✗ 无法读取图片: {image_file.name}")
                continue
            
            # 将BGR转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用数据增强
            augmented = transform(image=image)
            augmented_image = augmented['image']
            
            # 将numpy数组转回PIL Image
            augmented_pil = Image.fromarray(augmented_image)
            
            # 保存到目标目录（同名）
            dst_file = Path(dst_dir) / image_file.name
            augmented_pil.save(dst_file, quality=95)
            
            print(f"✓ 已处理: {image_file.name} ({i}/{len(image_files)})")
            
        except Exception as e:
            print(f"✗ 处理失败: {image_file.name} - {str(e)}")
    
    print(f"\n数据增强完成！共处理 {len(image_files)} 张图片")
    print(f"结果保存在: {dst_dir}")

if __name__ == "__main__":
    # 源目录和目标目录
    SOURCE_DIR = "/mnt/data4/dcr/大模型网关/人脸识别/src/FaceSafety/Input/20251230_01"
    DESTINATION_DIR = "/mnt/data4/dcr/大模型网关/人脸识别/src/FaceSafety/Input/20251230_test03"
    
    augment_face_images_albumentations(SOURCE_DIR, DESTINATION_DIR)
